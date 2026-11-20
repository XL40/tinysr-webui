// TinySR WebUI 服务
// API:
//   GET  /          -> web/index.html
//   GET  /api/info  -> {"vulkan_capable": bool, "scale": 4, "tile": 64, "overlap": 8, "threads": n}
//   POST /api/sr    -> multipart: image=<file>, backend=<vulkan|cpu>
//                      返回 image/png + X-Elapsed-Ms header
#include <httplib.h>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image.h"
#include "stb_image_write.h"

#include "engine.hpp"

#include "gpu.h"  // ncnn::get_gpu_count()

#include <chrono>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>
#if defined(_WIN32)
#include <windows.h>  // GetModuleFileNameA (httplib 已含 winsock2, CMake 里定义 WIN32_LEAN_AND_MEAN)
#endif

using namespace tinysr;

#if defined(_WIN32)
// Vulkan 预检 (仅 Windows): 手动加载 vulkan-1.dll 并试创建 instance。
// 无 ICD / 驱动损坏的机器上 vkCreateInstance 直接失败 —— 此时跳过 ncnn 探测走 CPU,
// 不给 ncnn 的失败路径在 MSVC 运行时上表现的机会 (无 GPU CI runner 上实测进程
// 死在 ncnn 探测之后, Linux 同条件则正常优雅降级)。不依赖 vulkan 头文件,
// 手写的最小结构体布局与 vulkan_core.h 的 VkInstanceCreateInfo 一致 (x64)。
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
static int g_phase = 0;     // TINYSR_PHASE: 网格相位偏移 (latent 单位)
static int g_win = 0;       // TINYSR_WIN: 融合窗 (0/1 互补余弦坡 / 2 Hann)
static int g_resid = 1;     // TINYSR_RESID: 精确残差回注 (fp32, CPU)
static int g_resid_r = 4;   // TINYSR_RESID_R: 残差低通盒式半径 (原图域 px)
static float g_resid_alpha = 1.0f;  // TINYSR_RESID_ALPHA: 残差强度
static int g_dcfix = 1;     // TINYSR_DCFIX: 逐 tile DC 对齐 (消 tile 间亮度台阶)
static int g_tonefix = 1;   // TINYSR_TONEFIX: 亮度分布匹配 (消整体曝光偏移)
static int g_ocm = 0;       // TINYSR_OCM: overlap color matching (与画布重叠区 mean/std 对齐)
static int g_mb = 0;        // TINYSR_MB: multiband blending (低频宽窗/高频窄窗)

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
  cfg.phase = g_phase;
  cfg.win = g_win;
  cfg.resid = g_resid;
  cfg.resid_r = g_resid_r;
  cfg.resid_alpha = g_resid_alpha;
  cfg.dcfix = g_dcfix;
  cfg.tonefix = g_tonefix;
  cfg.ocm = g_ocm;
  cfg.mb = g_mb;
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

int main(int argc, char** argv) {
  int port = 8014;
  if (argc > 1) port = atoi(argv[1]);
  if (const char* env = getenv("TINYSR_MODELS")) g_models_dir = env;
  if (const char* env = getenv("TINYSR_FP16")) g_fp16 = atoi(env);
  if (const char* env = getenv("TINYSR_THREADS")) g_threads = atoi(env);
  if (const char* env = getenv("TINYSR_OVERLAP")) g_overlap = atoi(env);
  if (const char* env = getenv("TINYSR_PHASE")) g_phase = atoi(env);
  if (const char* env = getenv("TINYSR_WIN")) g_win = atoi(env);
  if (const char* env = getenv("TINYSR_RESID")) g_resid = atoi(env);
  if (const char* env = getenv("TINYSR_RESID_R")) g_resid_r = atoi(env);
  if (const char* env = getenv("TINYSR_RESID_ALPHA")) g_resid_alpha = atof(env);
  if (const char* env = getenv("TINYSR_DCFIX")) g_dcfix = atoi(env);
  if (const char* env = getenv("TINYSR_TONEFIX")) g_tonefix = atoi(env);
  if (const char* env = getenv("TINYSR_OCM")) g_ocm = atoi(env);
  if (const char* env = getenv("TINYSR_MB")) g_mb = atoi(env);

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
    char buf[384];
    snprintf(buf, sizeof(buf),
             R"({"vulkan_capable": %s, "scale": 4, "tile": 512, "step": %d, "overlap": %d, "phase": %d, "win": %d, "resid": %d, "threads": %u, "dcfix": %d, "tonefix": %d, "ocm": %d, "mb": %d})",
             gpu_ok ? "true" : "false", (64 - g_overlap) * 8, g_overlap, g_phase, g_win, g_resid,
             (unsigned)(g_threads > 0 ? g_threads : (int)std::thread::hardware_concurrency()),
             g_dcfix, g_tonefix, g_ocm, g_mb);
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

    int w, h, ch;
    uint8_t* rgb = stbi_load_from_memory(
        (const stbi_uc*)img_file.content.data(), (int)img_file.content.size(), &w, &h, &ch, 3);
    if (!rgb) {
      res.status = 400;
      res.set_content(R"({"error":"decode failed"})", "application/json");
      return;
    }
    fprintf(stdout, "[sr] input %dx%d backend=%s\n", w, h, vulkan ? "vulkan" : "cpu");

    std::vector<uint8_t> out;
    int ow, oh;
    const auto t0 = std::chrono::steady_clock::now();
    {
      // 模型加载与推理都串行化: 切换后端会重建模型实例
      std::lock_guard<std::mutex> lk(g_engine_mutex);
      if (!ensure_engine(vulkan)) {
        stbi_image_free(rgb);
        res.status = 500;
        res.set_content(R"({"error":"model load failed"})", "application/json");
        return;
      }
      if (!g_engine->upscale(rgb, w, h, 4, nullptr, out, ow, oh)) {
        stbi_image_free(rgb);
        res.status = 500;
        res.set_content(R"({"error":"inference failed"})", "application/json");
        return;
      }
    }
    const double ms = std::chrono::duration<double, std::milli>(
                          std::chrono::steady_clock::now() - t0)
                          .count();
    stbi_image_free(rgb);
    fprintf(stdout, "[sr] done %dx%d in %.1fms\n", ow, oh, ms);

    int png_size = 0;
    uint8_t* png = stbi_write_png_to_mem(out.data(), ow * 3, ow, oh, 3, &png_size);
    if (!png) {
      res.status = 500;
      return;
    }
    char hdr[64];
    snprintf(hdr, sizeof(hdr), "%.1f", ms);
    res.set_header("X-Elapsed-Ms", hdr);
    res.set_header("Access-Control-Expose-Headers", "X-Elapsed-Ms");
    res.set_content(reinterpret_cast<const char*>(png), (size_t)png_size, "image/png");
    STBIW_FREE(png);
  });

  fprintf(stderr, "[sr] boot: handlers ready\n");
#ifndef _WIN32
  // glibc: 行缓冲让重定向 stdout 也能实时输出; UCRT 不支持行缓冲 (_IOLBF=_IOFBF)
  // 且对 (NULL, 0) 参数组合走 invalid parameter handler -> Release 下 __fastfail
  // 无声死亡 (CI 无 GPU runner 实测), 所以 Windows 上跳过。
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
