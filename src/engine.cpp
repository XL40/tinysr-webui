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
// 因此没有任何像素的被除数接近 0, 且过渡宽度 = 整个重叠区。
// edge_lo / edge_hi: 该 tile 的这条边是否贴图像边界 (贴边时不衰减, 因为对面没有 tile)。
std::vector<float> axis_window(int t, int overlap, bool edge_lo, bool edge_hi) {
  std::vector<float> w((size_t)t);
  for (int u = 0; u < t; u++) {
    auto fade = [](float x) { return ramp01(x); };
    const float a = edge_lo ? 1.f : fade((u + 0.5f) / (float)overlap);
    const float b = edge_hi ? 1.f : fade((t - u - 0.5f) / (float)overlap);
    w[(size_t)u] = std::min(a, b);
  }
  return w;
}

struct TileGrid {
  // 成员按声明顺序初始化, step 必须在 rows/cols 之前
  int lh, lw, step, rows, cols;
  std::vector<int> ys, xs;   // tile 起点 (latent)
  TileGrid(int latent_h, int latent_w, int overlap)
      : lh(latent_h), lw(latent_w), step(kTile - overlap), rows(1), cols(1) {
    auto starts = [&](int n) {
      std::vector<int> v;
      const int last = n - kTile;
      for (int raw = 0; raw <= last; raw += step) v.push_back(raw);
      if (v.empty() || v.back() + kTile < n) v.push_back(std::max(last, 0));
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
  // Extractor::input 只保存浅引用: 输入 Mat 必须活到 extract() 之后
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
  dcfix_ = cfg.dcfix;
  tonefix_ = cfg.tonefix;
  flatgate_ = cfg.flatgate;
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

// ---- flatgate 辅助: 证据图 (LR 域 defocus 检测) + 频段金字塔 ----

// 单通道 float 双线性采样 (坐标 clamp 到边界)
static inline float bilin_s(const std::vector<float>& v, int w, int h, float y, float x) {
  y = std::clamp(y, 0.f, (float)(h - 1));
  x = std::clamp(x, 0.f, (float)(w - 1));
  const int yi = (int)y, xi = (int)x;
  const int yi1 = std::min(yi + 1, h - 1), xi1 = std::min(xi + 1, w - 1);
  const float fy = y - yi, fx = x - xi;
  return v[(size_t)yi * w + xi] * (1 - fx) * (1 - fy) + v[(size_t)yi * w + xi1] * fx * (1 - fy) +
         v[(size_t)yi1 * w + xi] * (1 - fx) * fy + v[(size_t)yi1 * w + xi1] * fx * fy;
}

// 2x 降采样: box blur r=1 后抽奇偶 (与原型 box2d+[::2,::2] 同口径)
static void down2_ch(const std::vector<float>& in, std::vector<float>& out, int h, int w,
                     std::vector<float>& tmp) {
  tmp.assign(in.size(), 0.f);
  box_blur_chw(in, tmp, 1, h, w, 1);
  const int H = (h + 1) / 2, W = (w + 1) / 2;
  out.resize((size_t)H * W);
  for (int y = 0; y < H; y++)
    for (int x = 0; x < W; x++)
      out[(size_t)y * W + x] = tmp[(size_t)std::min(y * 2, h - 1) * w + std::min(x * 2, w - 1)];
}

// 双线性重采样到任意目标尺寸 (金字塔上采样用)
static void resample_ch(const std::vector<float>& in, std::vector<float>& out,
                        int h, int w, int H, int W) {
  out.resize((size_t)H * W);
  for (int y = 0; y < H; y++) {
    const float fy = (y + 0.5f) * (float)h / H - 0.5f;
    for (int x = 0; x < W; x++) {
      const float fx = (x + 0.5f) * (float)w / W - 0.5f;
      out[(size_t)y * W + x] = bilin_s(in, w, h, fy, fx);
    }
  }
}

bool Engine::upscale_f(const float* rgb01, int w, int h, int scale,
                       const std::function<void(int, int, int)>& progress,
                       std::vector<float>& out01, int& ow, int& oh, bool hdr) {
  if (!enc_) return false;

  // ---- 1. bicubic 放大 + pad 到 512 倍数 (全程 [0,1] fp32, 8/16-bit 量化只在 I/O 边界) ----
  const int nw = (w * scale / 8) * 8, nh = (h * scale / 8) * 8;  // 8 对齐 (模型要求)
  ow = w * scale; oh = h * scale;
  std::vector<float> bigf((size_t)nw * nh * 3);
  if (!stbir_resize_float_linear(rgb01, w, h, w * 3 * (int)sizeof(float),
                                 bigf.data(), nw, nh, nw * 3 * (int)sizeof(float), STBIR_RGB)) {
    fprintf(stderr, "[sr] upscale resize failed %dx%d -> %dx%d\n", w, h, nw, nh);
    return false;
  }
  const int pH = (nh + 511) / 512 * 512, pW = (nw + 511) / 512 * 512;
  std::vector<float> pixel((size_t)pH * pW * 3, 0.f);
  // 同一趟 bicubic 结果上统计: adain 的 style 参考 (lr_for_align) 的 mean/std,
  // 以及 tonefix 需要的 8-bit 亮度直方图
  double s_sum[3] = {0, 0, 0}, s_sq[3] = {0, 0, 0};
  std::vector<uint8_t> gluma((size_t)nw * nh);
  const double npx = (double)nw * nh;
  for (int y = 0; y < nh; y++)
    for (int x = 0; x < nw; x++) {
      const size_t bi = ((size_t)y * nw + x) * 3;
      for (int ch = 0; ch < 3; ch++) {
        const float v = bigf[bi + ch];
        pixel[((size_t)y * pW + x) * 3 + ch] = v * 2.f - 1.f;
        s_sum[ch] += v;
        s_sq[ch] += (double)v * v;
      }
      gluma[(size_t)y * nw + x] = (uint8_t)std::clamp(
          (int)std::lround((299.f * bigf[bi] + 587.f * bigf[bi + 1] + 114.f * bigf[bi + 2]) * 0.255f),
          0, 255);
    }
  double s_mean[3], s_std[3];
  for (int ch = 0; ch < 3; ch++) {
    s_mean[ch] = s_sum[ch] / npx;
    s_std[ch] = std::sqrt(std::max(0.0, s_sq[ch] / npx - s_mean[ch] * s_mean[ch]));
  }

  // ---- 1.5 flatgate: LR 域 defocus 证据图 (每 job 一次) ----
  // 在 LR 图上找清晰边缘证据 (脊线 NMS + 对比度阈值): 有证据 = 在焦, 保护模型细节;
  // 远离证据的平坦区 = 虚化/纯渐变, 对模型新增内容做频段抑制 + 预算钳制。
  // 相对判据 (有无证据) 而非绝对幅值: 活跃度幅值随退化程度整体平移, 不可跨图标定。
  fg_mo_.clear();
  if (flatgate_) {
    const int cell = 24, prof = 4, dilate = 2;
    std::vector<float> yl((size_t)w * h), gm((size_t)w * h);
    for (size_t i = 0; i < yl.size(); i++)
      yl[i] = (299.f * rgb01[i * 3 + 0] + 587.f * rgb01[i * 3 + 1] + 114.f * rgb01[i * 3 + 2]) / 1000.f;
    for (int yy = 0; yy < h; yy++) {
      const int y0 = std::max(yy - 1, 0), y1 = std::min(yy + 1, h - 1);
      for (int xx = 0; xx < w; xx++) {
        const int x0 = std::max(xx - 1, 0), x1 = std::min(xx + 1, w - 1);
        const float gx = (yl[(size_t)yy * w + x1] - yl[(size_t)yy * w + x0]) * 0.5f;
        const float gy = (yl[(size_t)y1 * w + xx] - yl[(size_t)y0 * w + xx]) * 0.5f;
        gm[(size_t)yy * w + xx] = std::sqrt(gx * gx + gy * gy);
      }
    }
    float thr = 0.02f;
    {
      std::vector<float> srt = gm;
      const size_t idx = srt.size() * 88 / 100;
      std::nth_element(srt.begin(), srt.begin() + idx, srt.end());
      thr = std::max(srt[idx], 0.02f);
    }
    const int GH = (h + cell - 1) / cell, GW = (w + cell - 1) / cell;
    std::vector<int> cnt((size_t)GH * GW, 0);
    size_t ncand = 0;
    for (float g : gm)
      if (g > thr) ncand++;
    const size_t stride = ncand > 60000 ? ncand / 60000 + 1 : 1;
    size_t seen = 0;
    for (int yy = 0; yy < h; yy++) {
      const int ya = std::max(yy - 1, 0), yb = std::min(yy + 1, h - 1);
      for (int xx = 0; xx < w; xx++) {
        const size_t i = (size_t)yy * w + xx;
        if (gm[i] <= thr) continue;
        if ((seen++ % stride) != 0) continue;
        const float g = gm[i];
        const int xa = std::max(xx - 1, 0), xb = std::min(xx + 1, w - 1);
        const float gx = (yl[(size_t)yy * w + xb] - yl[(size_t)yy * w + xa]) * 0.5f;
        const float gy = (yl[(size_t)yb * w + xx] - yl[(size_t)ya * w + xx]) * 0.5f;
        const float gn = std::max(g, 1e-6f);
        const float ddx = gx / gn, ddy = gy / gn;
        // 容差非极大抑制: 只取脊线附近 (profile 才居中)
        if (g < 0.9f * bilin_s(gm, w, h, yy + ddy, xx + ddx)) continue;
        if (g < 0.9f * bilin_s(gm, w, h, yy - ddy, xx - ddx)) continue;
        // 沿梯度方向 ±4px profile, 对比度足够才算证据 (滤 JPEG 噪声伪边)
        float ps[2 * 4 + 1], pmin = 1e9f, pmax = -1e9f;
        for (int k = 0; k <= 2 * prof; k++) {
          ps[k] = bilin_s(yl, w, h, yy + (k - prof) * ddy, xx + (k - prof) * ddx);
          pmin = std::min(pmin, ps[k]);
          pmax = std::max(pmax, ps[k]);
        }
        if (pmax - pmin <= 0.06f) continue;
        cnt[(size_t)std::min(yy / cell, GH - 1) * GW + std::min(xx / cell, GW - 1)]++;
      }
    }
    // 证据格 (>=4 样本) 膨胀 dilate 格: 证据附近的过渡带也算"在焦保护区"
    std::vector<float> grid((size_t)GH * GW);
    size_t n_ev = 0;
    for (size_t i = 0; i < grid.size(); i++) {
      grid[i] = cnt[i] >= 4 ? 1.f : 0.f;
      n_ev += cnt[i] >= 4;
    }
    {
      std::vector<float> gd;
      box_blur_chw(grid, gd, 1, GH, GW, dilate);
      for (size_t i = 0; i < grid.size(); i++) grid[i] = gd[i] > 0.01f ? 1.f : 0.f;
    }
    // 网格 -> 逐像素 M (0=保护 1=虚焦/平坦), 羽化。证据格取反: 有证据 => M=0。
    std::vector<float> mlr((size_t)w * h);
    for (int yy = 0; yy < h; yy++)
      for (int xx = 0; xx < w; xx++)
        mlr[(size_t)yy * w + xx] =
            1.f - grid[(size_t)std::min(yy / cell, GH - 1) * GW + std::min(xx / cell, GW - 1)];
    std::vector<float> mtmp;
    box_blur_chw(mlr, mtmp, 1, h, w, cell / 2);
    mlr.swap(mtmp);
    fg_mo_.assign((size_t)nw * nh, 0.f);
    if (!stbir_resize_float_linear(mlr.data(), w, h, w * (int)sizeof(float),
                                   fg_mo_.data(), nw, nh, nw * (int)sizeof(float), STBIR_1CHANNEL))
      fg_mo_.clear();
    fprintf(stderr, "[sr] flatgate evidence cells %zu/%d (%.1f%%)\n",
            n_ev, GH * GW, 100.0 * n_ev / ((double)GH * GW));
  }

  const int lt_h = pH / 8, lt_w = pW / 8;
  if (!upscale_pass(rgb01, w, h,
                    nw, nh, ow, oh, pW, pH, lt_h, lt_w, pixel, bigf, gluma, s_mean,
                    s_std, progress, out01, hdr))
    return false;
  std::vector<float>().swap(pixel);
  return true;
}

// 单遍: 网格构建 + encode/transformer/decode + 后处理 + 尺寸对齐。
bool Engine::upscale_pass(const float* rgb01,
                          int w, int h, int nw, int nh, int ow, int oh, int pW, int pH,
                          int lt_h, int lt_w, const std::vector<float>& pixel,
                          const std::vector<float>& bigf, const std::vector<uint8_t>& gluma,
                          const double* s_mean, const double* s_std,
                          const std::function<void(int, int, int)>& progress,
                          std::vector<float>& pass_out, bool hdr) {
  TileGrid grid(lt_h, lt_w, overlap_);
  const int ov = overlap_;
  const double npx = (double)nw * nh;
  fprintf(stderr,
          "[sr] grid %dx%d (step %d, overlap %d), latent %dx%d\n",
          grid.cols, grid.rows, grid.step, ov, lt_w, lt_h);

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
      const auto wy = axis_window(kTile, ov, y == 0, y + kTile >= lt_h);
      const auto wx = axis_window(kTile, ov, x == 0, x + kTile >= lt_w);
      for (int ch = 0; ch < 16; ch++)
        for (int yy = 0; yy < kTile; yy++)
          for (int xx = 0; xx < kTile; xx++) {
            const float g = wy[yy] * wx[xx];
            latent[(((size_t)ch * lt_h + y + yy) * lt_w + x + xx)] += l[((size_t)ch * 64 + yy) * 64 + xx] * g;
            if (ch == 0) contrib[(size_t)(y + yy) * lt_w + x + xx] += g;
          }
      if (dump_tile("enc", r * grid.cols + c, l))
        fprintf(stderr, "[tile] enc %d %d %d %d\n", r * grid.cols + c, r, c, 0);
      if (progress) progress(0, ++done, 3 * grid.count());
    }
  for (size_t i = 0; i < latent.size(); i++) latent[i] /= contrib[i % ((size_t)lt_h * lt_w)];
  dump_tensor("latent.bin", latent);

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
      std::vector<float> pred;
      int oc, ohh, oww;
      if (!run_net(*tr_, tile.data(), 16, kTile, kTile, pred, oc, ohh, oww)) return false;
      const auto wy = axis_window(kTile, ov, y == 0, y + kTile >= lt_h);
      const auto wx = axis_window(kTile, ov, x == 0, x + kTile >= lt_w);
      for (int ch = 0; ch < 16; ch++)
        for (int yy = 0; yy < kTile; yy++)
          for (int xx = 0; xx < kTile; xx++)
            hr[(((size_t)ch * lt_h + y + yy) * lt_w + x + xx)] +=
                pred[((size_t)ch * 64 + yy) * 64 + xx] * wy[yy] * wx[xx];
      if (progress) progress(1, grid.count() + ++done, 3 * grid.count());
    }
  for (size_t i = 0; i < hr.size(); i++) hr[i] /= contrib[i % ((size_t)lt_h * lt_w)];
  dump_tensor("hr.bin", hr);
  std::vector<float>().swap(latent);
  std::vector<float>().swap(contrib);

  // ---- 4. decode: 64x64 latent -> 512px 窗口, 重叠区余弦融合 ----
  std::vector<float> img((size_t)pH * pW * 3, 0.f), contrib2((size_t)pH * pW, 0.f);
  float off_sum = 0.f, off_max = 0.f;       // dcfix 统计
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
      // 全局注意力使每个 tile 的输出亮度随内容漂移 (暗部呈拼接块)。
      // 在融合权重恒为 1 的核心区统计与 bicubic 参考的平均差, 整块平移后融合。
      if (dcfix_) {
        const int ly0 = (y == 0) ? 0 : ov, ly1 = (y + kTile >= lt_h) ? kTile : kTile - ov;
        const int lx0 = (x == 0) ? 0 : ov, lx1 = (x + kTile >= lt_w) ? kTile : kTile - ov;
        // bigf 是未 pad 的 nw×nh: 底/右边缘 tile 的采样窗必须裁进范围, 否则越界读
        const int py_end = std::min(nh, (y + ly1) * 8);
        const int px_end = std::min(nw, (x + lx1) * 8);
        double acc[3] = {0, 0, 0};
        long n = 0;
        for (int py = y * 8 + ly0 * 8; py < py_end; py++)
          for (int px = x * 8 + lx0 * 8; px < px_end; px++) {
            const float* g = &bigf[((size_t)py * nw + px) * 3];
            const int yy = py - y * 8, xx = px - x * 8;
            for (int ch = 0; ch < 3; ch++)
              acc[ch] += g[ch] -
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
      const auto wy = axis_window(kTile, ov, y == 0, y + kTile >= lt_h);
      const auto wx = axis_window(kTile, ov, x == 0, x + kTile >= lt_w);
      // [-1,1] 域单频带余弦融合
      for (int yy = 0; yy < 512; yy++)
        for (int xx = 0; xx < 512; xx++) {
          const float g = wy[yy / 8] * wx[xx / 8];  // 512px 窗 = latent 窗 x8
          const size_t oy = (size_t)(y * 8 + yy) * pW + (x * 8 + xx);
          for (int ch = 0; ch < 3; ch++) img[oy * 3 + ch] += d[((size_t)ch * 512 + yy) * 512 + xx] * g;
          contrib2[oy] += g;
        }
      if (progress) progress(2, 2 * grid.count() + ++done, 3 * grid.count());
    }
  if (dcfix_ && grid.count() > 0)
    fprintf(stderr, "[sr] dcfix |offset| mean %.2f/255 max %.2f/255\n",
            off_sum / (grid.count() * 3) * 255.f, off_max * 255.f);
  std::vector<float>().swap(hr);

  // ---- 5. 裁剪 + 归一化到 [0,1] + 尺寸对齐 ----
  std::vector<float> img2((size_t)nh * nw * 3);
  for (int y = 0; y < nh; y++)
    for (int x = 0; x < nw; x++) {
      const size_t si = ((size_t)y * pW + x);
      for (int ch = 0; ch < 3; ch++) {
        const float p = std::clamp(img[si * 3 + ch] / contrib2[si], -1.f, 1.f) / 2.f + 0.5f;
        img2[((size_t)y * nw + x) * 3 + ch] = p;
      }
    }
  std::vector<float>().swap(img);
  std::vector<float>().swap(contrib2);

  dump_tensor("img2.bin", img2);
  dump_tensor("bigf.bin", bigf);

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
  // 单步蒸馏模型对暗部有系统性提亮, adain/直方图匹配只约束边缘分布, 修不了这种
  // 条件偏置。guide 与输出逐像素对齐, 可直接估计条件均值
  //   bias[g] = E[ sr_luma - guide_luma | guide_luma = g ]
  // 从输出里减掉, 等价于按原图曝光校准的单调修正曲线。
  // 8-bit 直方图是 SDR 统计, HDR (PQ/HLG 码域) 跳过。
  if (tonefix_ && !hdr) {
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

  // ---- 6c. 精确残差回注 (residfix) ----
  // tile 伪影 (DC 漂移/台阶) 在低频带 (周期 >= ~24px): R = 原图 - BOX下采样(输出)
  // 是只含低频伪影的残差, 低通后上采样加回 => 低频带精确对齐原图, SR 高频不动。
  // 全程 fp32: 8bit 中间量化会丢亚 1/255 的台阶 (色彩断层)。
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
      const float* src = rgb01 + ((size_t)y * gw) * 3;
      for (int x = 0; x < gw; x++)
        for (int ch = 0; ch < 3; ch++)
          Rc[((size_t)ch * gh + y) * gw + x] =
              src[x * 3 + ch] - D[((size_t)y * gw + x) * 3 + ch];
    }
    std::vector<float>().swap(D);
    // c) σ=resid_r_ 三遍盒式高斯: 只留 tile 级校正量, 滤掉原图 JPEG 块噪声 (8px 周期)
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

  // ---- 6d. 细节预算门控 (flatgate) ----
  // 虚化/bokeh 是拍摄内容而非退化, 但模型会在近平坦梯度上发明纹理 (光环/布纹)。
  // 对模型新增细节 D = img2 - bigf 做逐像素预算限幅:
  //   |D| 允许到 B = kAmp·act + kFloor (act = 源图局部活跃度, box12 平滑)
  // 超出预算的成分软性压回 (smoothstep 拐点), 预算内结构原样保留。
  // 相对预算而非绝对阈值: 活跃度幅值跨图不可比, 绝对阈值会全域误杀重压缩软图。
  if (flatgate_) {
    // 无证据平坦区的频段抑制 (OBP): D 的 3 层 Laplacian 金字塔,
    // 低频全保留, 中频保留 (1-0.6M), 高频保留 (1-0.9M)。
    // bigf 锚点不被模糊 —— 原生虚焦已存在于 bigf, 只处理模型新增内容。
    if (!fg_mo_.empty()) {
      const float lam_m = 0.6f, lam_h = 0.9f;
      std::vector<float> d0((size_t)nh * nw), d1, d2, d3, up, mid, b2u;
      for (int ch = 0; ch < 3; ch++) {
        for (size_t i = 0; i < d0.size(); i++) d0[i] = img2[(size_t)i * 3 + ch];
        down2_ch(d0, d1, nh, nw, up);
        const int h1 = (nh + 1) / 2, w1 = (nw + 1) / 2;
        down2_ch(d1, d2, h1, w1, up);
        const int h2 = (h1 + 1) / 2, w2 = (w1 + 1) / 2;
        down2_ch(d2, d3, h2, w2, up);
        const int h3 = (h2 + 1) / 2, w3 = (w2 + 1) / 2;
        resample_ch(d1, up, h1, w1, nh, nw);              // b0 = d0 - up(d1): 高频
        for (size_t i = 0; i < d0.size(); i++) d0[i] -= up[i];
        resample_ch(d2, up, h2, w2, h1, w1);              // b1 = d1 - up(d2): 中频低段
        for (size_t i = 0; i < d1.size(); i++) d1[i] -= up[i];
        resample_ch(d3, up, h3, w3, h2, w2);              // b2 = d2 - up(d3): 中频高段
        for (size_t i = 0; i < d2.size(); i++) d2[i] -= up[i];
        resample_ch(d3, up, h3, w3, nh, nw);              // up => 低频 (全保留)
        resample_ch(d1, mid, h1, w1, nh, nw);             // mid => 中频合成
        resample_ch(d2, b2u, h2, w2, nh, nw);
        for (size_t i = 0; i < mid.size(); i++) mid[i] += b2u[i];
        for (size_t i = 0; i < d0.size(); i++) {
          const float m = fg_mo_[i];
          img2[(size_t)i * 3 + ch] = up[i] + (1 - lam_m * m) * mid[i] + (1 - lam_h * m) * d0[i];
        }
      }
    }
    const float kAmp = 4.f, kFloor = 1.5f / 255.f, kKnee = 1.6f;
    std::vector<float> y((size_t)nw * nh), act, asmooth, dsum((size_t)nw * nh), dsm;
    for (size_t i = 0; i < y.size(); i++)
      y[i] = (299.f * bigf[i * 3 + 0] + 587.f * bigf[i * 3 + 1] +
              114.f * bigf[i * 3 + 2]) / 1000.f;
    box_blur_chw(y, act, 1, nh, nw, 3);              // 高频能量 (7px 窗)
    for (size_t i = 0; i < act.size(); i++) act[i] = std::fabs(act[i] - y[i]);
    box_blur_chw(act, asmooth, 1, nh, nw, 12);       // 源活跃度, 平滑窗保证预算稳定
    for (int yy = 0; yy < nh; yy++)
      for (int xx = 0; xx < nw; xx++) {
        const size_t i = (size_t)yy * nw + xx;
        float s = 0;
        for (int ch = 0; ch < 3; ch++) {
          const float d = img2[i * 3 + ch] - bigf[i * 3 + ch];
          s += d * d;
        }
        dsum[i] = std::sqrt(s / 3.f);
      }
    box_blur_chw(dsum, dsm, 1, nh, nw, 12);          // 模型加细节的幅度场
    double gsum = 0;
    int nclip = 0;
    for (int yy = 0; yy < nh; yy++)
      for (int xx = 0; xx < nw; xx++) {
        const size_t i = (size_t)yy * nw + xx;
        const float B = kAmp * asmooth[i] + kFloor;
        float g = 1.f;
        if (dsm[i] > B && dsm[i] > 0.f) {
          const float t = std::clamp((dsm[i] - B) / (kKnee * B - B), 0.f, 1.f);
          const float ts = t * t * (3.f - 2.f * t);  // smoothstep 拐点: 预算内不动
          g = (1.f - ts) + ts * (B / dsm[i]);
          nclip++;
        }
        gsum += g;
        if (g < 1.f)
          for (int ch = 0; ch < 3; ch++) {
            const float ref = bigf[i * 3 + ch];
            img2[i * 3 + ch] = ref + g * (img2[i * 3 + ch] - ref);
          }
      }
    fprintf(stderr, "[sr] flatgate budget mean gain %.3f, clipped px %.1f%%\n",
            gsum / ((double)nw * nh), 100.0 * nclip / ((double)nw * nh));
  }

  // ---- 7. 尺寸对齐 (nw,nh -> ow,oh, 双三次) + 本遍输出 (fp32, 量化在 I/O 边界) ----
  pass_out.assign((size_t)ow * oh * 3, 0.f);
  if (ow == nw && oh == nh) {
    for (size_t i = 0; i < pass_out.size(); i++)
      pass_out[i] = std::clamp(img2[i], 0.f, 1.f);
  } else {
    // 注意: float 版本的 stride 单位是字节 (像素数 * 4)
    const int istride = nw * 3 * (int)sizeof(float), ostride = ow * 3 * (int)sizeof(float);
    if (!stbir_resize_float_linear(img2.data(), nw, nh, istride, pass_out.data(), ow, oh, ostride, STBIR_RGB)) {
      fprintf(stderr, "[sr] resize failed %dx%d -> %dx%d\n", nw, nh, ow, oh);
      return false;
    }
    for (size_t i = 0; i < pass_out.size(); i++)
      pass_out[i] = std::clamp(pass_out[i], 0.f, 1.f);
  }
  return true;
}

}  // namespace tinysr
