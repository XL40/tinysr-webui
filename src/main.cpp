// TinySR WebUI 服务
// API:
//   GET  /            -> web/index.html
//   GET  /api/info    -> 能力 (vulkan/threads/格式开关等)
//   GET  /api/progress-> 当前任务进度
//   POST /api/sr      -> multipart: image=<file>, backend=<vulkan|cpu>, scale=, fmt=
//                        返回 image/* (fmt 决定) + X-Elapsed-Ms 等 header
//   POST /api/probe   -> multipart: image=<file>, 返回 {w,h,bits,hdr,fmt}
#include <httplib.h>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image.h"
#include "stb_image_write.h"

#include "engine.hpp"

#include "gpu.h"  // ncnn::get_gpu_count()

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cctype>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

// ---- 可选图像格式 (系统库, 均为免版税/开源许可; CMake 探测, 缺库则该格式禁用) ----
// stb 内置: JPEG / PNG / BMP / GIF(首帧) / TGA / PNM / PSD
// 外接: WebP(libwebp) TIFF(libtiff) AVIF 8/10/12-bit 含 HDR(libavif) DNG(LibRaw)
// HEIC/HEIF 不做: HEVC 专利池收费, 与"只做免版税格式"原则冲突, 同用途用 AVIF。
#if defined(TINYSR_HAVE_WEBP)
#include <webp/decode.h>
#include <webp/encode.h>
#endif
#if defined(TINYSR_HAVE_TIFF)
#include <tiffio.h>
#endif
#if defined(TINYSR_HAVE_AVIF)
#include <avif/avif.h>
#endif
#if defined(TINYSR_HAVE_DNG)
#include <libraw/libraw.h>
#endif
#if defined(_WIN32)
#include <windows.h>  // GetModuleFileNameA (httplib 已含 winsock2, CMake 里定义 WIN32_LEAN_AND_MEAN)
#endif

using namespace tinysr;

#if defined(_WIN32)
// Vulkan 预检 (仅 Windows): 手动加载 vulkan-1.dll 并试创建 instance。
// 无 ICD / 驱动损坏时直接失败 -> 跳过 ncnn 探测走 CPU (ncnn 的失败路径在 MSVC
// 运行时上会硬崩, Linux 则优雅降级)。不依赖 vulkan 头文件, 手写的最小结构体
// 布局与 vulkan_core.h 的 VkInstanceCreateInfo 一致 (x64)。
static bool vulkan_instance_ok() {
  HMODULE h = LoadLibraryA("vulkan-1.dll");
  if (!h) return false;
  using GetProcFn = void* (*)(void*, const char*);
  const auto get_proc = reinterpret_cast<GetProcFn>(
      reinterpret_cast<void*>(GetProcAddress(h, "vkGetInstanceProcAddr")));
  if (!get_proc) { FreeLibrary(h); return false; }
  using CreateFn = long (*)(const void*, const void*, void**);
  const auto vk_create = reinterpret_cast<CreateFn>(get_proc(nullptr, "vkCreateInstance"));
  if (!vk_create) { FreeLibrary(h); return false; }
  struct {
    unsigned sType;        // VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO = 1
    const void* pNext;
    unsigned flags;
    const void* pApplicationInfo;
    unsigned enabledLayerCount;
    const void* ppEnabledLayerNames;
    unsigned enabledExtensionCount;
    const void* ppEnabledExtensionNames;
  } ci = {1u, nullptr, 0u, nullptr, 0u, nullptr, 0u, nullptr};
  void* inst = nullptr;
  const long vr = vk_create(&ci, nullptr, &inst);
  fprintf(stderr, "[sr] vulkan preflight: vkCreateInstance -> %ld\n", vr);
  if (vr == 0 /*VK_SUCCESS*/ && inst) {
    const auto destroy = reinterpret_cast<void (*)(void*, const void*)>(
        get_proc(inst, "vkDestroyInstance"));
    if (destroy) destroy(inst, nullptr);
    FreeLibrary(h);
    return true;
  }
  FreeLibrary(h);
  return false;
}
#endif

static std::mutex g_engine_mutex;  // 单模型实例, 一次处理一张
static std::unique_ptr<Engine> g_engine;
static bool g_engine_vulkan = false;
static std::string g_models_dir = "models";
static int g_fp16 = 2;   // TINYSR_FP16: 0=off 1=storage 2=full
static int g_threads = 0;   // TINYSR_THREADS: 0=ncnn 默认
static int g_overlap = 8;   // TINYSR_OVERLAP: tile 重叠 (latent 单位)
static int g_resid = 1;     // TINYSR_RESID: 精确残差回注 (fp32, CPU)
static int g_resid_r = 4;   // TINYSR_RESID_R: 残差低通盒式半径 (原图域 px)
static float g_resid_alpha = 1.0f;  // TINYSR_RESID_ALPHA: 残差强度
static int g_dcfix = 1;     // TINYSR_DCFIX: 逐 tile DC 对齐 (消 tile 间亮度台阶)
static int g_tonefix = 1;   // TINYSR_TONEFIX: 亮度分布匹配 (消整体曝光偏移)
static int g_flatgate = 1;  // TINYSR_FLATGATE: 平坦区细节门控 (防虚化区幻觉纹理)
static int g_scale = 4;     // TINYSR_SCALE: 默认放大倍数 (请求可带 scale= 覆盖)
static bool g_verbose = false;      // TINYSR_VERBOSE / --verbose: 逐 tile 进度打 stderr
// /api/progress 用的任务状态 (单任务服务: 引擎串行, 新任务覆盖旧值)
static std::atomic<int64_t> g_job{0};
static std::atomic<int> g_stage{-1}, g_done{0}, g_total{0};
static std::atomic<int64_t> g_job_ms{0};   // 任务开始时刻 (steady_clock ms)

static std::string g_exe_dir;  // 可执行文件所在目录 (用于在任意 cwd 下找到 web/ 和 models/)

static bool file_exists(const std::string& p) {
  if (FILE* f = fopen(p.c_str(), "rb")) {
    fclose(f);
    return true;
  }
  return false;
}

static bool ensure_engine(bool vulkan) {
  if (g_engine && g_engine_vulkan == vulkan) return true;
  // 切换后端时先释放旧实例, 避免两套 1.4GB 权重同时在内存里
  g_engine.reset();
  auto e = std::make_unique<Engine>();
  Config cfg;
  cfg.models_dir = g_models_dir;
  cfg.vulkan = vulkan;
  cfg.fp16 = g_fp16;
  cfg.threads = g_threads;
  cfg.overlap = g_overlap;
  cfg.resid = g_resid;
  cfg.resid_r = g_resid_r;
  cfg.resid_alpha = g_resid_alpha;
  cfg.dcfix = g_dcfix;
  cfg.tonefix = g_tonefix;
  cfg.flatgate = g_flatgate;
  if (!e->init(cfg)) return false;
  g_engine = std::move(e);
  g_engine_vulkan = vulkan;
  fprintf(stdout, "[engine] loaded backend: %s\n", vulkan ? "vulkan" : "cpu");
  return true;
}

static std::string read_file(const std::string& path) {
  FILE* f = fopen(path.c_str(), "rb");
  if (!f) return "";
  std::string s;
  char buf[65536];
  size_t n;
  while ((n = fread(buf, 1, sizeof(buf), f)) > 0) s.append(buf, n);
  fclose(f);
  return s;
}

// 依次尝试候选路径, 返回第一个能读到的内容
static std::string read_first(const std::vector<std::string>& paths) {
  for (const auto& p : paths) {
    auto s = read_file(p);
    if (!s.empty()) return s;
  }
  return "";
}

// ============ 图像解码链 (输入) ============
// 顺序: stb 内置 -> WebP -> AVIF -> DNG(LibRaw) -> TIFF。
// 统一输出 [0,1] fp32 RGB; HDR (AVIF 的 PQ/HLG) 保持在感知码域, 位深原样保留。
struct DecodedImage {
  std::vector<float> rgb01;
  int w = 0, h = 0;
  int bits = 8;                // 源位深 (8/10/12/16)
  bool hdr = false;            // PQ / HLG (AVIF NCLX 判定)
  std::string fmt;
};

static bool dec_stb(const uint8_t* d, size_t n, DecodedImage& out) {
  // 不能用 stbi_loadf: float API 会做 sRGB->linear 转换, 而管线全程在码值域。
  // 16-bit PNG 在此处降到 8-bit, 高位深源走 AVIF/TIFF/DNG 路径
  int w, h, ch;
  uint8_t* p = stbi_load_from_memory((const stbi_uc*)d, (int)n, &w, &h, &ch, 3);
  if (!p) return false;
  out.w = w; out.h = h; out.bits = 8; out.hdr = false; out.fmt = "stb";
  out.rgb01.resize((size_t)w * h * 3);
  for (size_t i = 0; i < out.rgb01.size(); i++) out.rgb01[i] = p[i] / 255.f;
  stbi_image_free(p);
  return true;
}

#if defined(TINYSR_HAVE_WEBP)
static bool dec_webp(const uint8_t* d, size_t n, DecodedImage& out) {
  if (n < 12 || memcmp(d, "RIFF", 4) != 0 || memcmp(d + 8, "WEBP", 4) != 0) return false;
  int w, h;
  if (!WebPGetInfo(d, n, &w, &h)) return false;
  uint8_t* p = WebPDecodeRGB(d, n, &w, &h);
  if (!p) return false;
  out.w = w; out.h = h; out.bits = 8; out.hdr = false; out.fmt = "webp";
  out.rgb01.resize((size_t)w * h * 3);
  for (size_t i = 0; i < out.rgb01.size(); i++) out.rgb01[i] = p[i] / 255.f;
  WebPFree(p);
  return true;
}
#endif

#if defined(TINYSR_HAVE_AVIF)
static bool dec_avif(const uint8_t* d, size_t n, DecodedImage& out) {
  if (n < 12 || memcmp(d + 4, "ftyp", 4) != 0) return false;
  avifDecoder* dec = avifDecoderCreate();
  if (avifDecoderSetIOMemory(dec, d, n) != AVIF_RESULT_OK ||
      avifDecoderParse(dec) != AVIF_RESULT_OK || avifDecoderNextImage(dec) != AVIF_RESULT_OK) {
    avifDecoderDestroy(dec);
    return false;
  }
  avifImage* im = dec->image;
  avifRGBImage rgb;
  avifRGBImageSetDefaults(&rgb, im);
  rgb.format = AVIF_RGB_FORMAT_RGB;
  rgb.chromaUpsampling = AVIF_CHROMA_UPSAMPLING_BEST_QUALITY;
  if (avifRGBImageAllocatePixels(&rgb) != AVIF_RESULT_OK) {
    avifDecoderDestroy(dec);
    return false;
  }
  const avifResult rr = avifImageYUVToRGB(im, &rgb);
  if (rr != AVIF_RESULT_OK) {
    avifRGBImageFreePixels(&rgb);
    avifDecoderDestroy(dec);
    return false;
  }
  out.w = (int)im->width; out.h = (int)im->height;
  out.bits = (int)im->depth;
  out.hdr = im->transferCharacteristics == AVIF_TRANSFER_CHARACTERISTICS_SMPTE2084 ||
            im->transferCharacteristics == AVIF_TRANSFER_CHARACTERISTICS_HLG;
  out.fmt = "avif";
  out.rgb01.resize((size_t)out.w * out.h * 3);
  const float full = (float)((1 << out.bits) - 1);
  if (out.bits > 8) {
    const uint16_t* s = (const uint16_t*)rgb.pixels;
    for (size_t i = 0; i < out.rgb01.size(); i++) out.rgb01[i] = s[i] / full;
  } else {
    for (size_t i = 0; i < out.rgb01.size(); i++) out.rgb01[i] = rgb.pixels[i] / 255.f;
  }
  avifRGBImageFreePixels(&rgb);
  avifDecoderDestroy(dec);
  return true;
}
#endif

#if defined(TINYSR_HAVE_DNG)
// 仅 DNG (Adobe 开放规范, 免版税): LibRaw 做 CFA 去马赛克 + 相机白平衡, 输出 sRGB 16-bit。
static bool dec_dng(const uint8_t* d, size_t n, DecodedImage& out) {
  LibRaw rr;
  rr.imgdata.params.output_bps = 16;
  rr.imgdata.params.no_auto_bright = 1;
  rr.imgdata.params.use_camera_wb = 1;
  if (rr.open_buffer((void*)d, n) != LIBRAW_SUCCESS) return false;
  if (rr.unpack() != LIBRAW_SUCCESS || rr.dcraw_process() != LIBRAW_SUCCESS) return false;
  int err = 0;
  libraw_processed_image_t* im = rr.dcraw_make_mem_image(&err);
  if (!im || im->colors != 3 || im->bits != 16) {
    if (im) LibRaw::dcraw_clear_mem(im);
    return false;
  }
  out.w = im->width; out.h = im->height;
  out.bits = 16; out.hdr = false; out.fmt = "dng";
  out.rgb01.resize((size_t)im->width * im->height * 3);
  const float inv = 1.f / 65535.f;
  const uint16_t* s = (const uint16_t*)im->data;
  for (size_t i = 0; i < out.rgb01.size(); i++) out.rgb01[i] = s[i] * inv;
  LibRaw::dcraw_clear_mem(im);
  return true;
}
#endif

#if defined(TINYSR_HAVE_TIFF)
namespace {
struct TiffMem { const uint8_t* d; toff_t n, off; };
tmsize_t t_read(thandle_t h, tdata_t b, tmsize_t n) {
  TiffMem* m = (TiffMem*)h;
  if (m->off + (toff_t)n > m->n) return 0;
  memcpy(b, m->d + m->off, (size_t)n);
  m->off += (toff_t)n;
  return n;
}
tmsize_t t_write(thandle_t, tdata_t, tmsize_t) { return -1; }
toff_t t_seek(thandle_t h, toff_t off, int whence) {
  TiffMem* m = (TiffMem*)h;
  if (whence == SEEK_SET) m->off = off;
  else if (whence == SEEK_CUR) m->off += off;
  else m->off = m->n;
  return m->off;
}
int t_close(thandle_t) { return 0; }
toff_t t_size(thandle_t h) { return ((TiffMem*)h)->n; }
int t_map(thandle_t, tdata_t*, toff_t*) { return 0; }
void t_unmap(thandle_t, tdata_t, toff_t) {}
}  // namespace

// 灰度/RGB 8-bit 走 RGBA 接口 (自动处理压缩与调色板); 16-bit 逐行读保精度。
static bool dec_tiff(const uint8_t* d, size_t n, DecodedImage& out) {
  if (n < 8 || (memcmp(d, "II", 2) != 0 && memcmp(d, "MM", 2) != 0)) return false;
  TiffMem mem{d, (toff_t)n, 0};
  TIFF* t = TIFFClientOpen("mem", "r", &mem, t_read, t_write, t_seek, t_close, t_size, t_map, t_unmap);
  if (!t) return false;
  uint32_t w = 0, h = 0;
  uint16_t spp = 0, bps = 0, photo = 0, planar = PLANARCONFIG_CONTIG;
  TIFFGetField(t, TIFFTAG_IMAGEWIDTH, &w);
  TIFFGetField(t, TIFFTAG_IMAGELENGTH, &h);
  TIFFGetField(t, TIFFTAG_SAMPLESPERPIXEL, &spp);
  TIFFGetField(t, TIFFTAG_BITSPERSAMPLE, &bps);
  TIFFGetField(t, TIFFTAG_PHOTOMETRIC, &photo);
  TIFFGetField(t, TIFFTAG_PLANARCONFIG, &planar);
  if (!w || !h) { TIFFClose(t); return false; }
  out.w = (int)w; out.h = (int)h;
  if (bps == 16 && planar == PLANARCONFIG_CONTIG && (spp == 3 || spp == 1) &&
      (photo == PHOTOMETRIC_RGB || photo == PHOTOMETRIC_MINISBLACK)) {
    out.bits = 16; out.hdr = false; out.fmt = "tiff16";
    out.rgb01.resize((size_t)w * h * 3);
    tdata_t buf = _TIFFmalloc(TIFFScanlineSize(t));
    if (!buf) { TIFFClose(t); return false; }
    const float inv = 1.f / 65535.f;
    for (uint32_t row = 0; row < h; row++) {
      if (TIFFReadScanline(t, buf, row, 0) != 1) {
        _TIFFfree(buf); TIFFClose(t);
        return false;
      }
      const uint16_t* s = (const uint16_t*)buf;
      for (uint32_t x = 0; x < w; x++) {
        const size_t o = ((size_t)row * w + x) * 3;
        if (spp == 3) {
          out.rgb01[o + 0] = s[x * 3 + 0] * inv;
          out.rgb01[o + 1] = s[x * 3 + 1] * inv;
          out.rgb01[o + 2] = s[x * 3 + 2] * inv;
        } else {
          const float g = s[x] * inv;
          out.rgb01[o + 0] = out.rgb01[o + 1] = out.rgb01[o + 2] = g;
        }
      }
    }
    _TIFFfree(buf); TIFFClose(t);
    return true;
  }
  std::vector<uint32_t> rgba((size_t)w * h);
  if (!TIFFReadRGBAImageOriented(t, w, h, rgba.data(), ORIENTATION_TOPLEFT, 0)) {
    TIFFClose(t);
    return false;
  }
  TIFFClose(t);
  out.bits = 8; out.hdr = false; out.fmt = "tiff8";
  out.rgb01.resize((size_t)w * h * 3);
  for (size_t i = 0; i < (size_t)w * h; i++) {
    const uint8_t* p = (const uint8_t*)&rgba[i];   // RGBA 内存序
    out.rgb01[i * 3 + 0] = p[0] / 255.f;
    out.rgb01[i * 3 + 1] = p[1] / 255.f;
    out.rgb01[i * 3 + 2] = p[2] / 255.f;
  }
  return true;
}
#endif

static std::string lower_ext(const std::string& filename) {
  std::string e = std::filesystem::path(filename).extension().string();
  for (char& c : e) c = (char)tolower((unsigned char)c);
  return e;
}

static bool decode_any(const uint8_t* d, size_t n, const std::string& ext,
                       DecodedImage& out, std::string& err) {
#if defined(TINYSR_HAVE_DNG)
  // .dng 扩展名直接交给 LibRaw
  if (ext == ".dng" && dec_dng(d, n, out)) return true;
#endif
  if (dec_stb(d, n, out)) return true;
#if defined(TINYSR_HAVE_WEBP)
  if (dec_webp(d, n, out)) return true;
#endif
#if defined(TINYSR_HAVE_AVIF)
  if (dec_avif(d, n, out)) return true;
#endif
#if defined(TINYSR_HAVE_DNG)
  // TIFF magic 的相机文件 (扩展名可能不是 .dng) 也给 LibRaw 一次机会
  if (n >= 4 && (memcmp(d, "II", 2) == 0 || memcmp(d, "MM", 2) == 0) && dec_dng(d, n, out))
    return true;
#endif
#if defined(TINYSR_HAVE_TIFF)
  if (dec_tiff(d, n, out)) return true;
#endif
  err = "decode failed; supported: JPEG PNG BMP GIF(首帧) TGA PNM PSD"
#if defined(TINYSR_HAVE_WEBP)
        " WebP"
#endif
#if defined(TINYSR_HAVE_TIFF)
        " TIFF"
#endif
#if defined(TINYSR_HAVE_AVIF)
        " AVIF"
#endif
#if defined(TINYSR_HAVE_DNG)
        " DNG"
#endif
        " (HEIC/HEIF 不支持: HEVC 专利限制, 请用 AVIF)";
  return false;
}

// ============ 图像编码 (输出: png / jpg / bmp / webp / avif) ============
static void stb_collect(void* ctx, void* data, int n) {
  auto* v = (std::vector<uint8_t>*)ctx;
  v->insert(v->end(), (uint8_t*)data, (uint8_t*)data + n);
}

static std::vector<uint8_t> quant8(const std::vector<float>& v) {
  std::vector<uint8_t> q(v.size());
  for (size_t i = 0; i < v.size(); i++)
    q[i] = (uint8_t)std::clamp((int)std::lround(v[i] * 255.f), 0, 255);
  return q;
}

static bool enc_stb(const std::string& fmt, const std::vector<float>& out01, int w, int h,
                    std::vector<uint8_t>& body) {
  const std::vector<uint8_t> q = quant8(out01);
  if (fmt == "png")
    return stbi_write_png_to_func(stb_collect, &body, w, h, 3, q.data(), w * 3) != 0;
  if (fmt == "jpg")
    return stbi_write_jpg_to_func(stb_collect, &body, w, h, 3, q.data(), 95) != 0;
  if (fmt == "bmp")
    return stbi_write_bmp_to_func(stb_collect, &body, w, h, 3, q.data()) != 0;
  return false;
}

#if defined(TINYSR_HAVE_WEBP)
static bool enc_webp(const std::vector<float>& out01, int w, int h, std::vector<uint8_t>& body) {
  const std::vector<uint8_t> q = quant8(out01);
  uint8_t* p = nullptr;
  const size_t n = WebPEncodeRGB(q.data(), w, h, w * 3, 90.f, &p);
  if (!n || !p) return false;
  body.assign(p, p + n);
  WebPFree(p);
  return true;
}
#endif

#if defined(TINYSR_HAVE_AVIF)
// HDR 进 -> HDR 出: 保持源位深与 PQ/HLG 传递函数; SDR 输出 8-bit sRGB。
static bool enc_avif(const std::vector<float>& out01, int w, int h, int bits, bool hdr,
                     std::vector<uint8_t>& body) {
  avifImage* im = avifImageCreate(w, h, bits, AVIF_PIXEL_FORMAT_YUV444);
  if (!im) return false;
  im->colorPrimaries = hdr ? AVIF_COLOR_PRIMARIES_BT2020 : AVIF_COLOR_PRIMARIES_BT709;
  im->transferCharacteristics = hdr ? AVIF_TRANSFER_CHARACTERISTICS_SMPTE2084
                                    : AVIF_TRANSFER_CHARACTERISTICS_SRGB;
  im->matrixCoefficients = hdr ? AVIF_MATRIX_COEFFICIENTS_BT2020_NCL
                               : AVIF_MATRIX_COEFFICIENTS_BT601;
  im->alphaPremultiplied = AVIF_FALSE;
  avifRGBImage rgb;
  avifRGBImageSetDefaults(&rgb, im);
  rgb.format = AVIF_RGB_FORMAT_RGB;
  if (avifRGBImageAllocatePixels(&rgb) != AVIF_RESULT_OK) {
    avifImageDestroy(im);
    return false;
  }
  const float full = (float)((1 << bits) - 1);
  if (bits > 8) {
    uint16_t* p = (uint16_t*)rgb.pixels;
    for (size_t i = 0; i < out01.size(); i++)
      p[i] = (uint16_t)std::clamp((int)std::lround(out01[i] * full), 0, (1 << bits) - 1);
  } else {
    const std::vector<uint8_t> q = quant8(out01);
    memcpy(rgb.pixels, q.data(), q.size());
  }
  const avifResult cv = avifImageRGBToYUV(im, &rgb);
  avifRGBImageFreePixels(&rgb);
  if (cv != AVIF_RESULT_OK) {
    avifImageDestroy(im);
    return false;
  }
  avifEncoder* e = avifEncoderCreate();
  e->quality = 90;
  e->speed = 8;                       // 0=最慢最好 10=最快; 8 兼顾大图编码时间
  avifRWData rw = AVIF_DATA_EMPTY;
  const bool ok = avifEncoderAddImage(e, im, 1, 0) == AVIF_RESULT_OK &&
                  avifEncoderFinish(e, &rw) == AVIF_RESULT_OK;
  if (ok) body.assign(rw.data, rw.data + rw.size);
  avifRWDataFree(&rw);
  avifEncoderDestroy(e);
  avifImageDestroy(im);
  return ok;
}
#endif

static bool encode_any(const std::string& fmt, const std::vector<float>& out01, int w, int h,
                       int bits, bool hdr, std::vector<uint8_t>& body, const char*& ctype) {
  if (fmt == "png") { ctype = "image/png"; return enc_stb("png", out01, w, h, body); }
  if (fmt == "jpg") { ctype = "image/jpeg"; return enc_stb("jpg", out01, w, h, body); }
  if (fmt == "bmp") { ctype = "image/bmp"; return enc_stb("bmp", out01, w, h, body); }
#if defined(TINYSR_HAVE_WEBP)
  if (fmt == "webp") { ctype = "image/webp"; return enc_webp(out01, w, h, body); }
#endif
#if defined(TINYSR_HAVE_AVIF)
  if (fmt == "avif") { ctype = "image/avif"; return enc_avif(out01, w, h, bits, hdr, body); }
#endif
  return false;
}

int main(int argc, char** argv) {
  int port = 8014;
  bool port_set = false;
  for (int i = 1; i < argc; i++) {
    const std::string a = argv[i];
    if (a == "--verbose" || a == "-v") { g_verbose = true; continue; }
    if (!port_set) { port = atoi(a.c_str()); port_set = true; }
  }
  if (const char* env = getenv("TINYSR_MODELS")) g_models_dir = env;
  if (const char* env = getenv("TINYSR_FP16")) g_fp16 = atoi(env);
  if (const char* env = getenv("TINYSR_SCALE")) g_scale = std::clamp(atoi(env), 1, 8);
  if (const char* env = getenv("TINYSR_VERBOSE")) g_verbose = atoi(env) != 0;
  if (const char* env = getenv("TINYSR_THREADS")) g_threads = atoi(env);
  if (const char* env = getenv("TINYSR_OVERLAP")) g_overlap = atoi(env);
  if (const char* env = getenv("TINYSR_RESID")) g_resid = atoi(env);
  if (const char* env = getenv("TINYSR_RESID_R")) g_resid_r = atoi(env);
  if (const char* env = getenv("TINYSR_RESID_ALPHA")) g_resid_alpha = atof(env);
  if (const char* env = getenv("TINYSR_DCFIX")) g_dcfix = atoi(env);
  if (const char* env = getenv("TINYSR_TONEFIX")) g_tonefix = atoi(env);
  if (const char* env = getenv("TINYSR_FLATGATE")) g_flatgate = atoi(env);

  // 从任意工作目录启动都能找到 web/ 与 models/: 以二进制所在目录作为兜底
#if defined(_WIN32)
  char self_buf[MAX_PATH];
  const DWORD nself = GetModuleFileNameA(nullptr, self_buf, MAX_PATH);
  if (nself > 0 && nself < MAX_PATH) g_exe_dir = std::filesystem::path(self_buf).parent_path().string();
#else
  std::error_code ec;
  const auto self = std::filesystem::read_symlink("/proc/self/exe", ec);
  if (!ec) g_exe_dir = self.parent_path().string();
#endif

  // 没显式指定模型目录时, cwd 里找不到就试 ../models 和 exe 旁的 models
  if (!getenv("TINYSR_MODELS") && !file_exists(g_models_dir + "/vae_encoder.ncnn.param")) {
    std::vector<std::string> cands = { "../models", "models" };
    if (!g_exe_dir.empty()) {
      cands.push_back(g_exe_dir + "/../models");
      cands.push_back(g_exe_dir + "/models");
    }
    for (const auto& d : cands)
      if (file_exists(d + "/vae_encoder.ncnn.param")) { g_models_dir = d; break; }
  }

  httplib::Server svr;

  svr.Get("/", [](const httplib::Request&, httplib::Response& res) {
    std::vector<std::string> cands = { "web/index.html" };
    if (!g_exe_dir.empty()) {
      cands.push_back(g_exe_dir + "/web/index.html");
      cands.push_back(g_exe_dir + "/index.html");  // POST_BUILD 会把 web/ 拷到二进制旁
    }
    auto html = read_first(cands);
    if (html.empty()) {
      res.status = 404;
      res.set_content("index.html missing", "text/plain");
      return;
    }
    res.set_content(html, "text/html");
  });

  fprintf(stderr, "[sr] probing Vulkan...\n");
#if defined(_WIN32)
  const bool gpu_ok = vulkan_instance_ok() && ncnn::get_gpu_count() > 0;
#else
  const bool gpu_ok = ncnn::get_gpu_count() > 0;
#endif
  fprintf(stderr, "[sr] Vulkan probe done: %s\n", gpu_ok ? "GPU available" : "no device, CPU only");
  fprintf(stderr, "[sr] boot: registering handlers\n");
  svr.Get("/api/info", [gpu_ok](const httplib::Request&, httplib::Response& res) {
    const char* gpu_name = "cpu";
#if NCNN_VULKAN
    if (gpu_ok) {
      const ncnn::VulkanDevice* vkdev = ncnn::get_gpu_device(0);
      if (vkdev) gpu_name = vkdev->info.device_name();
    }
#endif
    char buf[640];
    snprintf(buf, sizeof(buf),
             R"({"vulkan_capable": %s, "gpu": "%s", "scale": %d, "tile": 512, "step": %d, "overlap": %d, "resid": %d, "threads": %u, "dcfix": %d, "tonefix": %d, "flatgate": %d, "webp": %s, "tiff": %s, "avif": %s, "dng": %s})",
             gpu_ok ? "true" : "false", gpu_name, g_scale, (64 - g_overlap) * 8, g_overlap, g_resid,
             (unsigned)(g_threads > 0 ? g_threads : (int)std::thread::hardware_concurrency()),
             g_dcfix, g_tonefix, g_flatgate,
#if defined(TINYSR_HAVE_WEBP)
             "true",
#else
             "false",
#endif
#if defined(TINYSR_HAVE_TIFF)
             "true",
#else
             "false",
#endif
#if defined(TINYSR_HAVE_AVIF)
             "true",
#else
             "false",
#endif
#if defined(TINYSR_HAVE_DNG)
             "true"
#else
             "false"
#endif
             );
    res.set_content(buf, "application/json");
  });

  svr.Post("/api/probe", [](const httplib::Request& req, httplib::Response& res) {
    // 浏览器解不出的格式 (DNG 等): 上传前探测尺寸, 前端用它显示输出预估
    auto img_file = req.form.get_file("image");
    if (img_file.content.empty()) {
      res.status = 400;
      res.set_content(R"({"error":"no image"})", "application/json");
      return;
    }
    DecodedImage img;
    std::string err;
    if (!decode_any((const uint8_t*)img_file.content.data(), img_file.content.size(),
                    lower_ext(img_file.filename), img, err)) {
      res.status = 400;
      res.set_content(std::string(R"({"error":")" ) + err + "\"}", "application/json");
      return;
    }
    char buf[192];
    snprintf(buf, sizeof(buf),
             R"({"w": %d, "h": %d, "bits": %d, "hdr": %s, "fmt": "%s"})",
             img.w, img.h, img.bits, img.hdr ? "true" : "false", img.fmt.c_str());
    res.set_content(buf, "application/json");
  });

  svr.Get("/api/progress", [](const httplib::Request&, httplib::Response& res) {
    // 单任务服务: 前端在上传期间轮询这个端点刷进度条
    const int64_t now_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                               std::chrono::steady_clock::now().time_since_epoch())
                               .count();
    char buf[256];
    snprintf(buf, sizeof(buf),
             R"({"job": %lld, "stage": %d, "done": %d, "total": %d, "elapsed_ms": %lld})",
             (long long)g_job.load(), g_stage.load(), g_done.load(), g_total.load(),
             (long long)std::max<int64_t>(0, now_ms - g_job_ms.load()));
    res.set_content(buf, "application/json");
  });

  svr.Post("/api/sr", [gpu_ok](const httplib::Request& req, httplib::Response& res) {
    auto img_file = req.form.get_file("image");
    if (img_file.content.empty()) {
      res.status = 400;
      res.set_content(R"({"error":"no image"})", "application/json");
      return;
    }
    const bool vulkan = gpu_ok && req.form.get_field("backend") != "cpu";
    int scale = g_scale;
    if (const std::string s = req.form.get_field("scale"); !s.empty())
      scale = std::clamp(atoi(s.c_str()), 1, 8);
    // 输出格式: png(默认)/jpg/bmp/webp/avif; webp/avif 需编译期带对应系统库
    std::string fmt = req.form.get_field("fmt");
    for (char& c : fmt) c = (char)tolower((unsigned char)c);
    if (fmt.empty()) fmt = "png";

    // ---- 解码 (多格式, 统一 [0,1] fp32) ----
    DecodedImage img;
    std::string derr;
    if (!decode_any((const uint8_t*)img_file.content.data(), img_file.content.size(),
                    lower_ext(img_file.filename), img, derr)) {
      res.status = 400;
      res.set_content(std::string(R"({"error":")" ) + derr + "\"}", "application/json");
      return;
    }
    fprintf(stdout, "[sr] input %s %dx%d bits=%d hdr=%d backend=%s scale=%d\n",
            img.fmt.c_str(), img.w, img.h, img.bits, img.hdr ? 1 : 0,
            vulkan ? "vulkan" : "cpu", scale);

    // ---- 进度上报 ----
    const int64_t job = ++g_job;
    g_stage.store(-1);
    g_done.store(0);
    g_total.store(0);
    g_job_ms.store(std::chrono::duration_cast<std::chrono::milliseconds>(
                       std::chrono::steady_clock::now().time_since_epoch())
                       .count());
    auto progress = [job](int stage, int done, int total) {
      if (g_job.load(std::memory_order_relaxed) != job) return;
      g_stage.store(stage);
      g_done.store(done);
      g_total.store(total);
      if (g_verbose) fprintf(stderr, "[sr] tile %d/%d (stage %d)\n", done, total, stage);
    };

    // ---- 推理 (HDR 跳过 SDR 专用校准) ----
    std::vector<float> out01;
    int ow, oh;
    const auto t0 = std::chrono::steady_clock::now();
    {
      // 模型加载与推理都串行化: 切换后端会重建模型实例
      std::lock_guard<std::mutex> lk(g_engine_mutex);
      if (!ensure_engine(vulkan)) {
        res.status = 500;
        res.set_content(R"({"error":"model load failed"})", "application/json");
        return;
      }
      if (!g_engine->upscale_f(img.rgb01.data(), img.w, img.h, scale, progress, out01,
                               ow, oh, img.hdr)) {
        res.status = 500;
        res.set_content(R"({"error":"inference failed"})", "application/json");
        return;
      }
      if (g_job.load() == job) progress(3, 0, 1);   // 后处理 (编码)
    }
    const double ms = std::chrono::duration<double, std::milli>(
                          std::chrono::steady_clock::now() - t0)
                          .count();
    fprintf(stdout, "[sr] done %dx%d in %.1fms\n", ow, oh, ms);

    // ---- 编码 (输出格式; HDR AVIF 保持源位深) ----
    std::vector<uint8_t> body;
    const char* ctype = "application/octet-stream";
    if (!encode_any(fmt, out01, ow, oh, img.hdr ? img.bits : 8, img.hdr, body, ctype)) {
      res.status = 400;
      res.set_content(R"({"error":"encode failed: unknown output format or missing optional lib"})", "application/json");
      return;
    }
    char hdr[64];
    snprintf(hdr, sizeof(hdr), "%.1f", ms);
    res.set_header("X-Elapsed-Ms", hdr);
    res.set_header("X-Bit-Depth", std::to_string(img.hdr && fmt == "avif" ? img.bits : 8));
    res.set_header("X-Hdr", img.hdr ? "1" : "0");
    res.set_header("Access-Control-Expose-Headers", "X-Elapsed-Ms, X-Bit-Depth, X-Hdr");
    res.set_content(reinterpret_cast<const char*>(body.data()), body.size(), ctype);
  });

  fprintf(stderr, "[sr] boot: handlers ready\n");
#ifndef _WIN32
  // glibc: 行缓冲让重定向 stdout 实时输出; UCRT 不支持且会 __fastfail, Windows 跳过
  setvbuf(stdout, nullptr, _IOLBF, 0);
#endif
  if (!gpu_ok) fprintf(stderr, "[warn] no Vulkan device found, only CPU backend available\n");
  fprintf(stderr, "[sr] boot: listening on 0.0.0.0:%d\n", port);
  fprintf(stdout, "TinySR WebUI  http://127.0.0.1:%d  (models: %s, vulkan: %s)\n", port,
          g_models_dir.c_str(), gpu_ok ? "yes" : "no");
  if (!svr.listen("0.0.0.0", port)) {
    fprintf(stderr, "listen failed\n");
    return 1;
  }
  return 0;
}
