// TinySR ncnn 推理引擎
// 语义对齐官方 PyTorch 实现 (已与 CPU fp32 金标准数值对拍):
//   bicubic 4x -> [-1,1] -> vae.encode(512px tile)
//   -> transformer(64 latent tile, IR 输出 = z - pred)
//   -> vae.decode(64 latent tile) -> adain 色彩对齐 -> 亮度分布匹配
// 三个阶段共用 64-latent tile 网格 + 余弦窗融合 (无网格缝)
// tile 间另有 DC 对齐: transformer 的全 tile 注意力会让每个 tile 的平均亮度
// 随内容漂移 ~0.5/255 (平滑/暗部区域可见 banding), 用 bicubic 参考逐 tile 锚定。
// 后端: Vulkan GPU (fp16 storage) / CPU, 由 ncnn opt 切换
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
  int phase = 0;                      // 网格相位偏移 (latent), 二次相位推理用
  int dcfix = 1;                      // 逐 tile DC 对齐 (消 tile 间亮度台阶)
  int tonefix = 1;                    // 亮度分布匹配 (消模型整体曝光偏移, 如暗部提亮)
  int ocm = 0;                        // overlap color matching: 与已写入画布在重叠区做 mean/std 对齐
  int mb = 0;                         // multiband blending: 低频宽窗缓变, 高频窄窗
  int win = 0;                        // 融合窗: 0/1=互补余弦坡 (数值等价) 2=Hann 剖面
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

  // rgb: RGB8, 行主序, w*3 每行
  // 成功后 out = RGB8 尺寸 (w*scale, h*scale)
  bool upscale(const uint8_t* rgb, int w, int h, int scale,
               const std::function<void(int stage, int done, int total)>& progress,
               std::vector<uint8_t>& out, int& ow, int& oh);

  bool vulkan_enabled() const { return vulkan_; }

 private:
  std::unique_ptr<Net> enc_, tr_, dec_;
  bool vulkan_ = false;
  int threads_ = 0;
  int overlap_ = 8;
  int phase_ = 0;
  int dcfix_ = 1;
  int tonefix_ = 1;
  int ocm_ = 0;
  int mb_ = 0;
  int win_ = 0;
  int resid_ = 1;
  int resid_r_ = 4;
  float resid_alpha_ = 1.0f;
};

}  // namespace tinysr
