// TinySR ncnn 推理引擎实现 (语义对齐官方 PyTorch 实现, 已数值对拍)
#include "engine.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <thread>

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include "stb_image.h"
#include "stb_image_resize2.h"

#include "net.h"

namespace tinysr {

// 调试: TINYSR_DUMP=<dir> 时把各阶段中间结果落盘 (与 PyTorch 参考实现对拍用)
// 调试: TINYSR_DUMP_TILES=<dir> 时把每个 tile 的独立输出落盘 (分析拼块一致性)
static int dump_tile(const char* name, int idx, const std::vector<float>& v) {
  const char* dir = getenv("TINYSR_DUMP_TILES");
  if (!dir) return 0;
  char path[1024];
  snprintf(path, sizeof(path), "%s/%s_%03d.bin", dir, name, idx);
  if (FILE* f = fopen(path, "wb")) {
    fwrite(v.data(), sizeof(float), v.size(), f);
    fclose(f);
    return 1;
  }
  return 0;
}

static void dump_tensor(const char* name, const std::vector<float>& v) {
  const char* dir = getenv("TINYSR_DUMP");
  if (!dir) return;
  std::string path = std::string(dir) + "/" + name;
  double sum = 0, sq = 0, mx = 0;
  for (float x : v) { sum += x; sq += (double)x * x; mx = std::max(mx, (double)std::fabs(x)); }
  const double n = v.empty() ? 1.0 : (double)v.size();
  if (FILE* f = fopen(path.c_str(), "wb")) {
    fwrite(v.data(), sizeof(float), v.size(), f);
    fclose(f);
    fprintf(stderr, "[dump] %s n=%zu mean=%.6f std=%.6f absmax=%.4g\n",
            path.c_str(), v.size(), sum / n, std::sqrt(std::max(0.0, sq / n - (sum / n) * (sum / n))), mx);
  }
}

constexpr int kTile = 64;   // latent tile (-> 512px tile)
// 重叠宽度由 Config::overlap 决定 (latent 单位), 网格步长 = kTile - overlap

// ncnn::Net 的 RAII 包装
struct Net {
  ncnn::Net net;
  Net() {
    net.opt.num_threads = std::max(1u, std::thread::hardware_concurrency());
  }
  bool load(const std::string& dir, const std::string& name, bool vulkan, int threads, int fp16) {
    net.opt.use_vulkan_compute = vulkan;
    if (threads > 0) net.opt.num_threads = threads;
    // fp16: 1 = 仅存储(权重/中间结果 fp16, 运算 fp32); 2 = 存储+运算
    net.opt.use_fp16_packed = vulkan && fp16 >= 1;
    net.opt.use_fp16_storage = vulkan && fp16 >= 1;
    net.opt.use_fp16_arithmetic = vulkan && fp16 >= 2;
    if (net.load_param((dir + "/" + name + ".ncnn.param").c_str()) != 0) return false;
    if (net.load_model((dir + "/" + name + ".ncnn.bin").c_str()) != 0) return false;
    return true;
  }
};

namespace {

// 余弦斜坡: v<=0 -> 0, v>=1 -> 1, 中间 0.5-0.5cos(pi v)
inline float kPi() { return 3.14159265358979f; }
inline float ramp01(float v) {
  if (v <= 0.f) return 0.f;
  if (v >= 1.f) return 1.f;
  return 0.5f - 0.5f * std::cos(kPi() * v);
}

// 一维融合权重 (长度 t)。重叠区用互补斜坡, 相邻两块在重叠区权重和恰为 1,
// 因此没有任何像素的被除数接近 0, 且过渡宽度 = 整个重叠区 (而不是窄高斯的中段)。
// edge_lo / edge_hi: 该 tile 的这条边是否贴图像边界 (贴边时不衰减, 因为对面没有 tile)。
// win: 0/1 = 互补余弦坡 (数值等价, 0 默认), 
//      2 = Hann 剖面 (半个 tile 宽度渐起渐落, 依赖 contrib 归一化)。
std::vector<float> axis_window(int t, int overlap, bool edge_lo, bool edge_hi, int win = 0) {
  std::vector<float> w((size_t)t);
  const float half = t * 0.5f;
  for (int u = 0; u < t; u++) {
    float a, b;
    if (win == 2) {
      // Hann 剖面: 从每条边起半个 tile 宽度渐起, 中部恒 1
      auto side = [&](float d, bool edge) {
        if (edge) return 1.f;
        return d >= half ? 1.f : 0.5f - 0.5f * std::cos(kPi() * (d + 0.5f) / half);
      };
      a = side(float(u), edge_lo);
      b = side(float(t - 1 - u), edge_hi);
      w[(size_t)u] = a * b;
      continue;
    }
    auto fade = [&](float x) {
      x = ramp01(x);
      return win == 1 ? 0.5f - 0.5f * std::cos(kPi() * x) : x;
    };
    a = edge_lo ? 1.f : fade((u + 0.5f) / (float)overlap);
    b = edge_hi ? 1.f : fade((t - u - 0.5f) / (float)overlap);
    w[(size_t)u] = std::min(a, b);
  }
  return w;
}

struct TileGrid {
  // 注意: 成员按声明顺序初始化, step 必须在 rows/cols 之前
  int lh, lw, step, rows, cols;
  std::vector<int> ys, xs;   // 去重后的 tile 起点 (latent)
  // phase: 网格相位偏移 (latent), 用于"偏移网格二次推理取平均"实验。
  // phase=0 时与旧实现完全一致; phase>0 时额外补一块贴 0 起点, 保证顶/左覆盖。
  TileGrid(int latent_h, int latent_w, int overlap, int phase = 0)
      : lh(latent_h), lw(latent_w), step(kTile - overlap), rows(1), cols(1) {
    auto starts = [&](int n) {
      std::vector<int> v;
      const int last = n - kTile;
      for (int raw = phase; raw <= last; raw += step) {
        v.push_back(raw);
        if (phase > 0 && raw == phase) v.push_back(0);   // 相位网格补首块 (覆盖 [0, phase))
      }
      if (v.empty() || v.back() + kTile < n) v.push_back(std::max(last, 0));
      std::sort(v.begin(), v.end());
      v.erase(std::unique(v.begin(), v.end()), v.end());
      return v;
    };
    ys = starts(latent_h);
    xs = starts(latent_w);
    rows = (int)ys.size();
    cols = (int)xs.size();
  }
  void pos(int r, int c, int& y, int& x) const {
    y = ys[r];
    x = xs[c];
  }
  int count() const { return rows * cols; }
};

// 把 ncnn Mat (w,h,c) 拷出为连续 CHW
void mat_to_chw(const ncnn::Mat& m, std::vector<float>& out) {
  out.resize((size_t)m.c * m.h * m.w);
  for (int c = 0; c < m.c; c++) {
    const float* src = m.channel(c);
    for (int y = 0; y < m.h; y++) {
      const float* row = src + (size_t)y * m.w;
      std::copy(row, row + m.w, out.data() + ((size_t)c * m.h + y) * m.w);
    }
  }
}

// 连续 CHW -> ncnn Mat (拷入)
ncnn::Mat chw_to_mat(const float* data, int c, int h, int w) {
  ncnn::Mat m(w, h, c);
  for (int ci = 0; ci < c; ci++) {
    float* dst = m.channel(ci);
    for (int y = 0; y < h; y++) {
      const float* row = data + ((size_t)ci * h + y) * w;
      std::copy(row, row + w, dst + (size_t)y * w);
    }
  }
  return m;
}

// CHW 布局的盒式模糊 (边缘复制), O(w*h*c) 滑窗实现, multiband 低频分离用
void box_blur_chw(const std::vector<float>& in, std::vector<float>& out, int c, int h, int w, int r) {
  out.assign(in.size(), 0.f);
  std::vector<float> tmp(in.size());
  const size_t plane = (size_t)h * w;
  for (int ch = 0; ch < c; ch++) {
    const float* src = in.data() + ch * plane;
    float* dst = tmp.data() + ch * plane;
    for (int y = 0; y < h; y++) {  // 水平
      const float* row = src + (size_t)y * w;
      float* out_row = dst + (size_t)y * w;
      double acc = 0;
      for (int k = -r; k <= r; k++) acc += row[std::min(w - 1, std::max(0, k))];
      for (int x = 0; x < w; x++) {
        out_row[x] = (float)(acc / (2 * r + 1));
        const int add = std::min(w - 1, x + r + 1), sub = std::max(0, x - r);
        acc += row[add] - row[sub];
      }
    }
  }
  for (int ch = 0; ch < c; ch++) {  // 垂直
    const float* src = tmp.data() + ch * plane;
    float* dst = out.data() + ch * plane;
    for (int x = 0; x < w; x++) {
      double acc = 0;
      for (int k = -r; k <= r; k++) acc += src[(size_t)std::min(h - 1, std::max(0, k)) * w + x];
      for (int y = 0; y < h; y++) {
        dst[(size_t)y * w + x] = (float)(acc / (2 * r + 1));
        const int add = std::min(h - 1, y + r + 1), sub = std::max(0, y - r);
        acc += src[(size_t)add * w + x] - src[(size_t)sub * w + x];
      }
    }
  }
}

bool run_net(Net& n, const float* in, int ic, int ih, int iw, std::vector<float>& out,
             int& oc, int& oh, int& ow) {
  ncnn::Extractor ex = n.net.create_extractor();
  // 注意: Extractor::input 只保存 Mat 的浅引用, 临时对象会在语句结束时析构 ->
  // 必须让输入 Mat 活到 extract() 之后 (同 Python binding 必须 .clone() 的坑)
  const ncnn::Mat in_mat = chw_to_mat(in, ic, ih, iw);
  ex.input("in0", in_mat);
  ncnn::Mat m;
  if (ex.extract("out0", m) != 0) return false;
  mat_to_chw(m, out);
  oc = m.c; oh = m.h; ow = m.w;
  return true;
}

}  // namespace

Engine::Engine() = default;
Engine::~Engine() = default;

bool Engine::init(const Config& cfg) {
  vulkan_ = cfg.vulkan;
  threads_ = cfg.threads;
  overlap_ = std::clamp(cfg.overlap, 2, kTile / 2);
  phase_ = std::clamp(cfg.phase, 0, kTile - 1);
  win_ = std::clamp(cfg.win, 0, 2);
  dcfix_ = cfg.dcfix;
  tonefix_ = cfg.tonefix;
  ocm_ = cfg.ocm;
  mb_ = cfg.mb;
  resid_ = cfg.resid;
  resid_r_ = std::clamp(cfg.resid_r, 1, 16);
  resid_alpha_ = cfg.resid_alpha;
  enc_ = std::make_unique<Net>();
  tr_ = std::make_unique<Net>();
  dec_ = std::make_unique<Net>();
  bool ok = enc_->load(cfg.models_dir, "vae_encoder", vulkan_, threads_, cfg.fp16) &&
            tr_->load(cfg.models_dir, "tinysr_transformer", vulkan_, threads_, cfg.fp16) &&
            dec_->load(cfg.models_dir, "vae_decoder", vulkan_, threads_, cfg.fp16);
  if (!ok) {
    enc_.reset(); tr_.reset(); dec_.reset();
    return false;
  }
  return true;
}

bool Engine::upscale(const uint8_t* rgb, int w, int h, int scale,
                     const std::function<void(int, int, int)>& progress,
                     std::vector<uint8_t>& out, int& ow, int& oh) {
  if (!enc_) return false;

  // ---- 1. bicubic 放大 + [-1,1] 归一化 + pad 到 512 倍数 ----
  const int nw = (w * scale / 8) * 8, nh = (h * scale / 8) * 8;  // 8 对齐 (模型要求)
  ow = w * scale; oh = h * scale;
  std::vector<uint8_t> big8((size_t)nw * nh * 3);
  if (!stbir_resize_uint8_linear(rgb, w, h, w * 3, big8.data(), nw, nh, nw * 3, STBIR_RGB)) {
    fprintf(stderr, "[sr] upscale resize failed %dx%d -> %dx%d\n", w, h, nw, nh);
    return false;
  }
  const int pH = (nh + 511) / 512 * 512, pW = (nw + 511) / 512 * 512;
  std::vector<float> pixel((size_t)pH * pW * 3, 0.f);
  // adain 的 style 参考 = bicubic 4x 放大后的 LR (即参考实现里的 lr_for_align),
  // 这里直接在同一批 bicubic 结果上统计, 省一次重采样
  // tonefix 还需要 bicubic 亮度的 256 级直方图 (LUT 的目标分布), 同一趟顺便统计
  double s_sum[3] = {0, 0, 0}, s_sq[3] = {0, 0, 0};
  // tonefix 需要 bicubic 参考的亮度 (uint8, 0..255) 做条件偏置修正
  std::vector<uint8_t> gluma((size_t)nw * nh);
  const double npx = (double)nw * nh;
  for (int y = 0; y < nh; y++)
    for (int x = 0; x < nw; x++)
      for (int ch = 0; ch < 3; ch++) {
        const float v = big8[((size_t)y * nw + x) * 3 + ch] / 255.f;
        pixel[((size_t)y * pW + x) * 3 + ch] = v * 2.f - 1.f;
        s_sum[ch] += v;
        s_sq[ch] += (double)v * v;
        if (ch == 2)
          gluma[(size_t)y * nw + x] =
              (uint8_t)((299 * big8[((size_t)y * nw + x) * 3 + 0] +
                         587 * big8[((size_t)y * nw + x) * 3 + 1] +
                         114 * big8[((size_t)y * nw + x) * 3 + 2]) / 1000);
      }
  double s_mean[3], s_std[3];
  for (int ch = 0; ch < 3; ch++) {
    s_mean[ch] = s_sum[ch] / npx;
    s_std[ch] = std::sqrt(std::max(0.0, s_sq[ch] / npx - s_mean[ch] * s_mean[ch]));
  }

  const int lt_h = pH / 8, lt_w = pW / 8;
  TileGrid grid(lt_h, lt_w, overlap_, phase_);
  const int ov = overlap_;
  fprintf(stderr, "[sr] grid %dx%d (step %d, overlap %d, phase %d), latent %dx%d\n",
          grid.cols, grid.rows, grid.step, ov, phase_, lt_w, lt_h);

  // ---- 2. encode: 512px 窗口 -> 64x64 latent, 重叠区余弦融合 ----
  std::vector<float> latent((size_t)lt_h * lt_w * 16, 0.f), contrib((size_t)lt_h * lt_w, 0.f);
  int done = 0;
  for (int r = 0; r < grid.rows; r++)
    for (int c = 0; c < grid.cols; c++) {
      int y, x; grid.pos(r, c, y, x);
      // pixel 窗口 [y*8, x*8] + 512 -> tile CHW (RGB planar)
      std::vector<float> tile((size_t)3 * 512 * 512);
      for (int ch = 0; ch < 3; ch++)
        for (int yy = 0; yy < 512; yy++)
          for (int xx = 0; xx < 512; xx++)
            tile[((size_t)ch * 512 + yy) * 512 + xx] =
                pixel[((size_t)(y * 8 + yy) * pW + (x * 8 + xx)) * 3 + ch];
      std::vector<float> l;
      int oc, ohh, oww;
      if (!run_net(*enc_, tile.data(), 3, 512, 512, l, oc, ohh, oww) || ohh != 64 || oww != 64)
        return false;
      const auto wy = axis_window(kTile, ov, y == 0, y + kTile >= lt_h, win_);
      const auto wx = axis_window(kTile, ov, x == 0, x + kTile >= lt_w, win_);
      for (int ch = 0; ch < 16; ch++)
        for (int yy = 0; yy < kTile; yy++)
          for (int xx = 0; xx < kTile; xx++) {
            const float g = wy[yy] * wx[xx];
            latent[(((size_t)ch * lt_h + y + yy) * lt_w + x + xx)] += l[((size_t)ch * 64 + yy) * 64 + xx] * g;
            if (ch == 0) contrib[(size_t)(y + yy) * lt_w + x + xx] += g;
          }
      if (dump_tile("enc", r * grid.cols + c, l))
        fprintf(stderr, "[tile] enc %d %d %d %d\n", r * grid.cols + c, r, c, 0);
      if (progress) progress(0, ++done, grid.count());
    }
  for (size_t i = 0; i < latent.size(); i++) latent[i] /= contrib[i % ((size_t)lt_h * lt_w)];
  dump_tensor("latent.bin", latent);
  std::vector<float>().swap(pixel);

  // ---- 3. transformer: 融合结果即 hr_latent (= z - pred) ----
  std::vector<float> hr((size_t)lt_h * lt_w * 16, 0.f);
  done = 0;
  for (int r = 0; r < grid.rows; r++)
    for (int c = 0; c < grid.cols; c++) {
      int y, x; grid.pos(r, c, y, x);
      std::vector<float> tile((size_t)16 * kTile * kTile);
      for (int ch = 0; ch < 16; ch++)
        for (int yy = 0; yy < kTile; yy++)
          std::copy(&latent[((size_t)ch * lt_h + y + yy) * lt_w + x],
                    &latent[((size_t)ch * lt_h + y + yy) * lt_w + x] + kTile,
                    &tile[((size_t)ch * kTile + yy) * kTile]);
      std::vector<float> p;
      int oc, ohh, oww;
      if (!run_net(*tr_, tile.data(), 16, kTile, kTile, p, oc, ohh, oww)) return false;
      const auto wy = axis_window(kTile, ov, y == 0, y + kTile >= lt_h, win_);
      const auto wx = axis_window(kTile, ov, x == 0, x + kTile >= lt_w, win_);
      for (int ch = 0; ch < 16; ch++)
        for (int yy = 0; yy < kTile; yy++)
          for (int xx = 0; xx < kTile; xx++)
            hr[(((size_t)ch * lt_h + y + yy) * lt_w + x + xx)] +=
                p[((size_t)ch * 64 + yy) * 64 + xx] * wy[yy] * wx[xx];
      if (progress) progress(1, ++done, grid.count());
    }
  for (size_t i = 0; i < hr.size(); i++) hr[i] /= contrib[i % ((size_t)lt_h * lt_w)];
  dump_tensor("hr.bin", hr);
  std::vector<float>().swap(latent);
  std::vector<float>().swap(contrib);

  // ---- 4. decode: 64x64 latent -> 512px 窗口, 重叠区余弦融合 ----
  std::vector<float> img((size_t)pH * pW * 3, 0.f), contrib2((size_t)pH * pW, 0.f);
  std::vector<float> imgL, cL;  // multiband 低频带 (宽窗)
  if (mb_) {
    imgL.assign((size_t)pH * pW * 3, 0.f);
    cL.assign((size_t)pH * pW, 0.f);
  }
  float off_sum = 0.f, off_max = 0.f;       // dcfix 统计
  double ocm_shift_sum = 0, ocm_gain_sum = 0;  // ocm 统计
  long ocm_n = 0;
  done = 0;
  for (int r = 0; r < grid.rows; r++)
    for (int c = 0; c < grid.cols; c++) {
      int y, x; grid.pos(r, c, y, x);
      std::vector<float> tile((size_t)16 * kTile * kTile);
      for (int ch = 0; ch < 16; ch++)
        for (int yy = 0; yy < kTile; yy++)
          std::copy(&hr[((size_t)ch * lt_h + y + yy) * lt_w + x],
                    &hr[((size_t)ch * lt_h + y + yy) * lt_w + x] + kTile,
                    &tile[((size_t)ch * kTile + yy) * kTile]);
      std::vector<float> d;
      int oc, ohh, oww;
      if (!run_net(*dec_, tile.data(), 16, kTile, kTile, d, oc, ohh, oww) || ohh != 512 || oww != 512)
        return false;
      // ---- 逐 tile DC 对齐 (dcfix) ----
      // transformer 对整块 64x64 latent 做全局注意力, 每个 tile 拿到的"全局语境"
      // 不同, 输出的平均亮度会随 tile 内容漂移 (实测平滑区相邻 tile 台阶中位
      // 0.5/255、p90 1.4/255, 暗部即肉眼可见的拼接块)。修法: 在该 tile 融合权重
      // 恒为 1 的核心区, 统计它与 bicubic 参考 (=原曝光, 无 tile 结构) 的平均差,
      // 整块平移后再融合。过渡仍由余弦窗完成, 不引入新台阶。
      if (dcfix_) {
        const int ly0 = (y == 0) ? 0 : ov, ly1 = (y + kTile >= lt_h) ? kTile : kTile - ov;
        const int lx0 = (x == 0) ? 0 : ov, lx1 = (x + kTile >= lt_w) ? kTile : kTile - ov;
        double acc[3] = {0, 0, 0};
        long n = 0;
        for (int yy = ly0 * 8; yy < ly1 * 8; yy++)
          for (int xx = lx0 * 8; xx < lx1 * 8; xx++) {
            const uint8_t* g = &big8[((size_t)(y * 8 + yy) * nw + (x * 8 + xx)) * 3];
            for (int ch = 0; ch < 3; ch++)
              acc[ch] += g[ch] / 255.f -
                         (d[((size_t)ch * 512 + yy) * 512 + xx] * 0.5f + 0.5f);
            n++;
          }
        for (int ch = 0; ch < 3; ch++) {
          // [0,1] 域的 offset; 限幅 ±15/255, 只修 DC 漂移不动正常内容
          const float off = (float)std::clamp(acc[ch] / (double)std::max(n, 1L), -0.06, 0.06);
          off_sum += std::fabs(off);
          off_max = std::max(off_max, std::fabs(off));
          for (int i = 0; i < 512 * 512; i++)
            d[(size_t)ch * 512 * 512 + i] += off * 2.f;  // [-1,1] 域
        }
      }
      if (dump_tile("dec", r * grid.cols + c, d))
        fprintf(stderr, "[tile] dec %d %d %d %d\n", r * grid.cols + c, r, c, 0);
      const auto wy = axis_window(kTile, ov, y == 0, y + kTile >= lt_h, win_);
      const auto wx = axis_window(kTile, ov, x == 0, x + kTile >= lt_w, win_);

      if (!ocm_ && !mb_) {
        // 基础路径: [-1,1] 域单频带余弦融合
        for (int yy = 0; yy < 512; yy++)
          for (int xx = 0; xx < 512; xx++) {
            const float g = wy[yy / 8] * wx[xx / 8];  // 512px 窗 = latent 窗 x8
            const size_t oy = (size_t)(y * 8 + yy) * pW + (x * 8 + xx);
            for (int ch = 0; ch < 3; ch++) img[oy * 3 + ch] += d[((size_t)ch * 512 + yy) * 512 + xx] * g;
            contrib2[oy] += g;
          }
      } else {
        // ocm / mb 路径: [0,1] 域
        std::vector<float> d01((size_t)3 * 512 * 512);
        for (size_t i = 0; i < d01.size(); i++) d01[i] = d[i] * 0.5f + 0.5f;

        // ---- overlap color matching (ocm) ----
        // 逐 tile 顺序累加时, 画布上已有左/上邻居的结果。在画布权重 >= 0.5
        // 的重叠条带里统计本 tile 与画布的逐通道 mean/std, 把整 tile 线性拉齐
        // (gain 限幅 [0.7,1.4]) 后再融合 -> 邻居间亮度/对比度漂移互相锚定。
        // 注意: 平坦区 (std < ~4/255) 的 std 比值纯是噪声, 必须 gain=1 跳过,
        // 否则一个歪 gain 会污染画布、被后续 tile 级联放大 (实测能到 150/255)。
        if (ocm_) {
          double st[3] = {0, 0, 0}, stq[3] = {0, 0, 0}, sa[3] = {0, 0, 0}, saq[3] = {0, 0, 0};
          long n = 0;
          for (int yy = 0; yy < 512; yy++) {
            const int oy_row = y * 8 + yy;
            for (int xx = 0; xx < 512; xx++) {
              const size_t oi = (size_t)oy_row * pW + (x * 8 + xx);
              if (contrib2[oi] < 0.5f) continue;  // 画布上邻居已主导的像素
              for (int ch = 0; ch < 3; ch++) {
                // mb 模式下画布 = 已累加的高带 + 低带; 否则就是完整信号
                const float a = mb_ ? img[oi * 3 + ch] / contrib2[oi] + imgL[oi * 3 + ch] / cL[oi]
                                    : img[oi * 3 + ch] / contrib2[oi];
                const float t = d01[((size_t)ch * 512 + yy) * 512 + xx];
                sa[ch] += a; saq[ch] += (double)a * a;
                st[ch] += t; stq[ch] += (double)t * t;
              }
              n++;
            }
          }
          if (n >= 2048) {
            float shift[3] = {0, 0, 0}, gain[3] = {1, 1, 1};
            for (int ch = 0; ch < 3; ch++) {
              const float mt = (float)(st[ch] / n), ma = (float)(sa[ch] / n);
              const float sst = std::sqrt(std::max(0.f, (float)(stq[ch] / n) - mt * mt));
              const float ssa = std::sqrt(std::max(0.f, (float)(saq[ch] / n) - ma * ma));
              // std 门限: 两侧都要有像样的纹理才做对比度匹配 (4/255)
              if (sst > 0.015f && ssa > 0.015f)
                gain[ch] = std::clamp(ssa / sst, 0.9f, 1.1f);
              shift[ch] = ma - mt * gain[ch];
              ocm_shift_sum += std::fabs(shift[ch]);
              ocm_gain_sum += std::fabs(gain[ch] - 1.f);
              ocm_n++;
            }
            for (int ch = 0; ch < 3; ch++)
              for (int i = 0; i < 512 * 512; i++) {
                float& v = d01[(size_t)ch * 512 * 512 + i];
                v = std::clamp(v * gain[ch] + shift[ch], 0.f, 1.f);
              }
          }
        }

        // ---- multiband blending (mb) ----
        // 低频 (盒式模糊 r=24px 分离) 用半宽 256px 的宽余弦窗缓变,
        // 高频保留原 64px 窄窗 -> tone 过渡拉宽, 纹理过渡保持锐利。
        const float* hi = d01.data();
        std::vector<float> dlow, dhi;
        if (mb_) {
          box_blur_chw(d01, dlow, 3, 512, 512, 24);
          dhi.resize(d01.size());
          for (size_t i = 0; i < d01.size(); i++) dhi[i] = d01[i] - dlow[i];
          hi = dhi.data();
          const auto wyw = axis_window(512, 256, y == 0, y + kTile >= lt_h, win_);
          const auto wxw = axis_window(512, 256, x == 0, x + kTile >= lt_w, win_);
          for (int yy = 0; yy < 512; yy++)
            for (int xx = 0; xx < 512; xx++) {
              const float g = wyw[yy] * wxw[xx];
              const size_t oy = (size_t)(y * 8 + yy) * pW + (x * 8 + xx);
              for (int ch = 0; ch < 3; ch++)
                imgL[oy * 3 + ch] += dlow[((size_t)ch * 512 + yy) * 512 + xx] * g;
              cL[oy] += g;
            }
        }
        // 高频 (或全信号): 窄窗, [0,1] 域累加
        for (int yy = 0; yy < 512; yy++)
          for (int xx = 0; xx < 512; xx++) {
            const float g = wy[yy / 8] * wx[xx / 8];
            const size_t oy = (size_t)(y * 8 + yy) * pW + (x * 8 + xx);
            for (int ch = 0; ch < 3; ch++) img[oy * 3 + ch] += hi[((size_t)ch * 512 + yy) * 512 + xx] * g;
            contrib2[oy] += g;
          }
      }
      if (progress) progress(2, ++done, grid.count());
    }
  if (dcfix_ && grid.count() > 0)
    fprintf(stderr, "[sr] dcfix |offset| mean %.2f/255 max %.2f/255\n",
            off_sum / (grid.count() * 3) * 255.f, off_max * 255.f);
  if (ocm_ && ocm_n > 0)
    fprintf(stderr, "[sr] ocm mean |shift| %.2f/255  mean |gain-1| %.3f (%d tiles)\n",
            ocm_shift_sum / ocm_n * 255.f, ocm_gain_sum / ocm_n, ocm_n / 3);
  std::vector<float>().swap(hr);
  std::vector<uint8_t>().swap(big8);

  // ---- 5. 裁剪 + 归一化到 [0,1] + 尺寸对齐 ----
  std::vector<float> img2((size_t)nh * nw * 3);
  for (int y = 0; y < nh; y++)
    for (int x = 0; x < nw; x++) {
      const size_t si = ((size_t)y * pW + x);
      for (int ch = 0; ch < 3; ch++) {
        float p;
        if (mb_)  // 低频宽窗带 + 高频窄窗带 (都在 [0,1] 域)
          p = std::clamp(img[si * 3 + ch] / contrib2[si] + imgL[si * 3 + ch] / cL[si], 0.f, 1.f);
        else if (ocm_)  // [0,1] 域单频带
          p = std::clamp(img[si * 3 + ch] / contrib2[si], 0.f, 1.f);
        else  // 基础路径: [-1,1] 域
          p = std::clamp(img[si * 3 + ch] / contrib2[si], -1.f, 1.f) / 2.f + 0.5f;
        img2[((size_t)y * nw + x) * 3 + ch] = p;
      }
    }
  std::vector<float>().swap(img);
  std::vector<float>().swap(contrib2);
  if (mb_) {
    std::vector<float>().swap(imgL);
    std::vector<float>().swap(cL);
  }

  dump_tensor("img2.bin", img2);

  // ---- 6. adain 色彩对齐 (作者 adain_color_fix: target=结果, style=bicubic LR) ----
  //     out = (target - mean_t) / std_t * std_s + mean_s   (逐通道标量)
  for (int ch = 0; ch < 3; ch++) {
    double mt = 0, t_sq = 0;
    for (size_t i = 0; i < (size_t)nw * nh; i++) {
      const double v = img2[i * 3 + ch];
      mt += v;
      t_sq += v * v;
    }
    mt /= npx;
    const double t_std = std::sqrt(std::max(0.0, t_sq / npx - mt * mt));
    const double gain = (t_std > 1e-6) ? s_std[ch] / t_std : 0.0;
    for (size_t i = 0; i < (size_t)nw * nh; i++)
      img2[i * 3 + ch] = (float)std::clamp(
          (img2[i * 3 + ch] - mt) * gain + s_mean[ch], 0.0, 1.0);
  }

  // ---- 6b. 条件偏置修正 (tonefix) ----
  // 单步蒸馏模型对暗部有系统性提亮 (实测 L20-40 层平均 +1.3/255, L<20 达
  // +8.8/255), adain 的全局均值/方差匹配和直方图匹配都修不了这种"条件偏置"
  // (它们只约束边缘分布)。guide 与输出逐像素对齐, 所以可以直接估计条件均值
  //   bias[g] = E[ sr_luma - guide_luma | guide_luma = g ]
  // 再从输出里减掉, 等价于按原图曝光校准的一条单调修正曲线, 不动彩色比例。
  if (tonefix_) {
    double cnt[256] = {0}, diff[256] = {0};
    const size_t npx2 = (size_t)nw * nh;
    for (size_t i = 0; i < npx2; i++) {
      const float l = 0.299f * img2[i * 3 + 0] + 0.587f * img2[i * 3 + 1] +
                      0.114f * img2[i * 3 + 2];
      const int g = gluma[i];
      const int s = std::clamp((int)std::lround(l * 255.f), 0, 255);
      cnt[g] += 1;
      diff[g] += s - g;
    }
    double bias[256];
    for (int g = 0; g < 256; g++) bias[g] = (cnt[g] > 0) ? diff[g] / cnt[g] : 0.0;
    // 平滑 (box r=6) 并填空档: 用邻域有效值的加权平均
    double sm[256];
    for (int g = 0; g < 256; g++) {
      double wsum = 0, vsum = 0;
      for (int k = std::max(0, g - 6); k <= std::min(255, g + 6); k++)
        if (cnt[k] > 0) { wsum += cnt[k]; vsum += cnt[k] * bias[k]; }
      sm[g] = (wsum > 0) ? vsum / wsum : 0.0;
    }
    double applied = 0;
    int napp = 0;
    for (size_t i = 0; i < npx2; i++) {
      const float l = 0.299f * img2[i * 3 + 0] + 0.587f * img2[i * 3 + 1] +
                      0.114f * img2[i * 3 + 2];
      const float b = (float)(sm[gluma[i]] / 255.0);  // [0,1] 域
      const float s = (l > 1e-4f) ? (l - b) / l : 1.f;
      for (int ch = 0; ch < 3; ch++)
        img2[i * 3 + ch] = std::clamp(img2[i * 3 + ch] * s, 0.f, 1.f);
      applied += std::fabs(b);
      napp++;
    }
    fprintf(stderr, "[sr] tonefix conditional bias: mean |shift| %.2f/255\n",
            applied / std::max(napp, 1) * 255.f);
  }

  // ---- 6c. 精确残差回注 (residfix, TINYSR_RESID, 全程 fp32) ----
  // tile 伪影 (DC 漂移/台阶/色调偏移) 全部生活在低频带 (周期 >= ~24px), 纹理细节
  // 在高频带: R = 原图 - BOX下采样(输出) 是只含低频伪影的精确残差, 低通后上采样
  // 加回 => 低频带精确对齐原图, SR 高频不受影响。8bit 中间量化会丢掉亚 1/255 的
  // 台阶信息 (色彩断层), 因此整条链 fp32, 只在最终写 PNG 时量化一次。
  if (resid_) {
    const int gw = w, gh = h;
    // a) BOX 下采样到原图尺寸 (均值保持的低通)
    std::vector<float> D((size_t)gw * gh * 3);
    {
      STBIR_RESIZE rs;
      stbir_resize_init(&rs, img2.data(), nw, nh, nw * 3 * (int)sizeof(float),
                        D.data(), gw, gh, gw * 3 * (int)sizeof(float),
                        STBIR_RGB, STBIR_TYPE_FLOAT);
      stbir_set_edgemodes(&rs, STBIR_EDGE_CLAMP, STBIR_EDGE_CLAMP);
      stbir_set_filters(&rs, STBIR_FILTER_BOX, STBIR_FILTER_BOX);
      if (!stbir_resize_extended(&rs)) {
        fprintf(stderr, "[sr] resid: downsample failed, skipped\n");
        return false;
      }
    }
    // b) R = 原图(fp32) - D, 打包成 CHW 供模糊
    std::vector<float> Rc((size_t)3 * gh * gw);
    for (int y = 0; y < gh; y++) {
      const uint8_t* src = rgb + ((size_t)y * gw) * 3;
      for (int x = 0; x < gw; x++)
        for (int ch = 0; ch < 3; ch++)
          Rc[((size_t)ch * gh + y) * gw + x] =
              src[x * 3 + ch] / 255.f - D[((size_t)y * gw + x) * 3 + ch];
    }
    std::vector<float>().swap(D);
    // c) σ=resid_r_ 三遍盒式高斯: 只留 tile 级 (周期 >= ~24px) 校正量,
    //    滤掉原图 JPEG 块噪声 (8px 周期) —— 实测 r<=2 时暗部可见注入噪声
    {
      std::vector<float> tmp;
      for (int p = 0; p < 3; p++) {
        box_blur_chw(Rc, tmp, 3, gh, gw, resid_r_);
        Rc.swap(tmp);
      }
    }
    // d) CHW -> HWC 后上采样回 nw×nh (残差带限, Catmull-Rom 足够) 并加回
    {
      std::vector<float> Rh((size_t)gw * gh * 3);
      for (int y = 0; y < gh; y++)
        for (int x = 0; x < gw; x++)
          for (int ch = 0; ch < 3; ch++)
            Rh[((size_t)y * gw + x) * 3 + ch] =
                Rc[((size_t)ch * gh + y) * gw + x];
      std::vector<float>().swap(Rc);
      std::vector<float> Rup((size_t)nw * nh * 3);
      STBIR_RESIZE rs;
      stbir_resize_init(&rs, Rh.data(), gw, gh, gw * 3 * (int)sizeof(float),
                        Rup.data(), nw, nh, nw * 3 * (int)sizeof(float),
                        STBIR_RGB, STBIR_TYPE_FLOAT);
      stbir_set_edgemodes(&rs, STBIR_EDGE_CLAMP, STBIR_EDGE_CLAMP);
      if (!stbir_resize_extended(&rs)) {
        fprintf(stderr, "[sr] resid: upsample failed, skipped\n");
        return false;
      }
      double acc = 0;
      long npx3 = (long)nw * nh * 3;
      for (long i = 0; i < npx3; i++) {
        const float corr = resid_alpha_ * Rup[i];
        acc += std::fabs(corr);
        img2[i] = std::clamp(img2[i] + corr, 0.f, 1.f);
      }
      fprintf(stderr, "[sr] resid |corr| mean %.2f/255 (sigma %d, alpha %.2f)\n",
              acc / npx3 * 255.f, resid_r_, resid_alpha_);
    }
  }

  // ---- 7. 尺寸对齐 (nw,nh -> ow,oh, 双三次) + 输出 ----
  out.resize((size_t)ow * oh * 3);
  if (ow == nw && oh == nh) {
    for (size_t i = 0; i < out.size(); i++)
      out[i] = (uint8_t)std::clamp((int)std::lround(img2[i] * 255.f), 0, 255);
  } else {
    std::vector<float> fin((size_t)ow * oh * 3);
    // 注意: float 版本的 stride 单位是字节 (像素数 * 4)
    const int istride = nw * 3 * (int)sizeof(float), ostride = ow * 3 * (int)sizeof(float);
    if (!stbir_resize_float_linear(img2.data(), nw, nh, istride, fin.data(), ow, oh, ostride, STBIR_RGB)) {
      fprintf(stderr, "[sr] resize failed %dx%d -> %dx%d\n", nw, nh, ow, oh);
      return false;
    }
    for (size_t i = 0; i < out.size(); i++)
      out[i] = (uint8_t)std::clamp((int)std::lround(std::clamp(fin[i], 0.f, 1.f) * 255.f), 0, 255);
  }
  return true;
}

}  // namespace tinysr
