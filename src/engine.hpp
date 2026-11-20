// TinySR ncnn 推理引擎
// 管线: bicubic 放大 -> [-1,1] -> vae.encode(512px tile)
//   -> transformer(64 latent tile, IR 输出 = z - pred)
//   -> vae.decode(64 latent tile) -> adain 色彩对齐 -> tonefix -> residfix -> flatgate
// 三阶段共用 latent tile 网格 + 余弦窗融合 (无网格缝); dcfix 逐 tile 锚定亮度。
// 后端: Vulkan GPU (fp16 storage) / CPU
#pragma once
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace tinysr {

struct Config {
  std::string models_dir = "models";  // tinysr_transformer / vae_encoder / vae_decoder (.ncnn.param/.bin)
  bool vulkan = true;                 // false = 纯 CPU
  int threads = 0;                    // 0 = auto (硬件并发)
  int fp16 = 2;                       // Vulkan 下: 0=关, 1=仅 fp16 存储, 2=存储+运算
  int overlap = 8;                    // tile 重叠 (latent 单位, 1 latent = 8 px)
  int dcfix = 1;                      // 逐 tile DC 对齐 (消 tile 间亮度台阶)
  int tonefix = 1;                    // 亮度分布匹配 (消模型整体曝光偏移, 如暗部提亮)
  int flatgate = 1;                   // 平坦区细节门控: 证据检测 + 频段抑制 + 预算钳制
  int resid = 1;                      // 精确残差回注 (低频带对齐原图, fp32)
  int resid_r = 4;                    // 残差低通: 盒式模糊半径 (原图域 px, 3 遍 ≈ σ=r)
  float resid_alpha = 1.0f;           // 残差强度
};

struct Net;  // engine.cpp 中定义 (ncnn::Net 的 RAII 包装)

class Engine {
 public:
  Engine();
  ~Engine();
  bool init(const Config& cfg);

  // 单一推理入口: 输入 [0,1] fp32 RGB (8/16-bit 源由调用方归一), 输出同域 RGB。
  // hdr=true: 源为 PQ/HLG 感知码域 —— 跳过 SDR 专用校准 (tonefix), 输出保持同域同位深。
  bool upscale_f(const float* rgb01, int w, int h, int scale,
                 const std::function<void(int stage, int done, int total)>& progress,
                 std::vector<float>& out01, int& ow, int& oh, bool hdr = false);

  bool vulkan_enabled() const { return vulkan_; }

 private:
  // 单遍推理: 网格构建 + encode/transformer/decode + 后处理 + 尺寸对齐。
  bool upscale_pass(const float* rgb01, int w,
                    int h, int nw, int nh, int ow, int oh, int pW, int pH, int lt_h,
                    int lt_w, const std::vector<float>& pixel,
                    const std::vector<float>& bigf, const std::vector<uint8_t>& gluma,
                    const double* s_mean, const double* s_std,
                    const std::function<void(int stage, int done, int total)>& progress,
                    std::vector<float>& pass_out, bool hdr = false);

  std::unique_ptr<Net> enc_, tr_, dec_;
  bool vulkan_ = false;
  int threads_ = 0;
  int overlap_ = 8;
  int dcfix_ = 1;
  int tonefix_ = 1;
  int flatgate_ = 1;
  std::vector<float> fg_mo_;   // flatgate: LR 域证据图上采样到画布尺寸 (0=保护 1=平坦)
  int resid_ = 1;
  int resid_r_ = 4;
  float resid_alpha_ = 1.0f;
};

}  // namespace tinysr
