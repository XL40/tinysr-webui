# TinySR WebUI

One-click upscaler for **[TinySR](https://github.com/Microtreei/TinySR)** (CVPR 2026 Findings) — a pruned single-step diffusion transformer (DiT) super-resolution model — running entirely on **ncnn** with **fp16**, on **Vulkan GPU** or **CPU**.

一条命令起一个网页，把任意图片 4× 超分。推理栈只有 ncnn，不依赖 PyTorch / diffusers / OpenVINO。

## 特性

- **单栈**：C++ + ncnn，无 Python 运行时依赖（导出模型时才需要 PyTorch）。
- **双后端**：网页里切换 Vulkan GPU（fp16 存储/运算）或 CPU（fp32）。
- **任意分辨率**：64-latent tile 网格 + 重叠区互补余弦加权融合，不限制输入尺寸。
- **拼接一致性**：逐 tile DC 对齐 + 曝光校准 + 精确残差回注，平滑/暗部区域无"图块感"。
- **fp16 而非 INT8**：移动/桌面 GPU 上 fp16 路径成熟且无需反量化，量化收益在这个模型上不成立（见下）。
- **自包含前端**：单个 `index.html`，拖放 → 上传 → 前后对比滑块 → 下载 PNG。

## 架构

```
浏览器 (拖放 / 滑块对比)
   │  multipart POST /api/sr  (image, backend=vulkan|cpu)
   ▼
cpp-httplib  (src/main.cpp)
   ▼
Engine::upscale  (src/engine.cpp)
   ├─ bicubic 4× → [-1,1] → pad 到 512 倍数
   ├─ vae_encoder   : 512px 窗口 → 16×64×64 latent   ┐
   ├─ transformer   : 16×64×64 → 16×64×64 (z−pred)    ├ 共用 64-latent tile 网格
   ├─ vae_decoder   : 16×64×64 → 512px 窗口           ┘ 重叠区余弦互补融合
   ├─ 逐 tile DC 对齐 + tonefix（消 tile 间亮度漂移 / 暗部系统性提亮）
   ├─ 精确残差回注 (residfix)：输出低频带对齐原图，全程 fp32
   └─ PNG (X-Elapsed-Ms 响应头带服务端耗时)
```

模型侧固定约束：`pos_embed` 为固定 1024 token（32×32 patch grid = 512px 像素 tile = 64×64 latent），所以任意分辨率必须 tile。

拼接一致性的问题归因、方案对比与全部实测数据见 **[docs/research.md](docs/research.md)**。

## 预编译包（Releases）

从 [Releases](../../releases) 下载对应平台 zip（可执行文件 + 前端），另下载 `models.zip` 解压到可执行文件旁的 `models/` 目录：

```
tinysr-webui.exe (或 tinysr-webui)
web/index.html
models/*.ncnn.param / *.ncnn.bin   ← 来自 models.zip
```

运行期依赖 GPU 驱动提供的 Vulkan loader（Windows: `vulkan-1.dll`，Linux: `libvulkan.so.1`，装了 NVIDIA/AMD/Intel 驱动即有）。纯 CPU 机器请源码 `-DTINYSR_VULKAN=OFF` 自建。

## 构建

依赖：CMake ≥ 3.14、C++17 编译器、Vulkan loader（`libvulkan.so`，GPU 后端）、系统里有 Vulkan 驱动。

```bash
# 1. 拉依赖 (ncnn 源码 + glslang 子模块 + 4 个单头库)
./tools/setup_third_party.sh
#    国内/受限网络: HTTPS_PROXY=http://127.0.0.1:PORT ./tools/setup_third_party.sh

# 2. 编译 (首次会编译 ncnn + glslang, 12 核约 3-5 分钟)
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build
```

只要 CPU 后端可关掉 Vulkan 省编译时间：`cmake -B build -DTINYSR_VULKAN=OFF`（前端会自动检测并禁用 GPU 选项）。

依赖版本在 `tools/setup_third_party.sh` 里钉死（ncnn `71f7318` / cpp-httplib `v0.66.0` / stb `9600522` / stb_image_resize2 `a4f303e`），并用 `HTTPS_PROXY` 支持代理；脚本会校验每个头文件的字节数与版本标记，防止拿到错误页或半截文件。

### Windows

同一条链路（PowerShell 里用 `cmake -B build` + `cmake --build build --config Release`）：

- 需要 Vulkan SDK（编译期链接 `vulkan-1.lib`）：[LunarG](https://vulkan.lunarg.com/sdk/home) 安装，或 `choco install vulkan-sdk`
- MSVC 下 CMake 自动加 `/utf-8`（源码含中文注释）、`NOMINMAX`，并静态链接运行时（用户机器免装 VC Redist）

### CI / 发布

- **ci.yml**：push/PR 时 Ubuntu + Windows 双矩阵编译 + 冒烟（起服务、`/api/info` 应答）
- **release.yml**：推 `v*` tag 触发，双平台预编译打包挂到 GitHub Release；模型权重不进 CI，发布后手动 `gh release upload <tag> models.zip`

## 模型

`models/` 下的 6 个文件（`*.ncnn.param` / `*.ncnn.bin`）需自行导出，不入库（权重 1.3GB）：

```bash
python tools/export_ncnn.py \
  --repo  /path/to/TinySR-main \
  --ckpt  /path/to/checkpoint \
  --out   models
```

脚本做三件事：合并 LoRA → `torch.jit.trace(check_trace=False)` → pnnx 转 ncnn。导出后与 torch fp32 对拍：三个模型 max abs diff ≤ 1.5e-5。

导出链路中的坑（已内建在脚本里）：

| 问题 | 处理 |
| --- | --- |
| `torch.jit.trace` 默认 `check_trace=True` 的 deepcopy 自校验对该图误报 | `check_trace=False`，数值正确性由导出后独立比对兜底 |
| pnnx 无法转换 6D `reshape/permute`（patchify 还原） | unpatchify 改写为**固定权重 `ConvTranspose2d(k=2,s=2)`**（PixelShuffle 等价），全程只有 4D |
| `ConvTranspose2d` 权重形状易搞反 | 形状为 `[in, out, kh, kw]`（与 `Conv2d` 相反） |
| 想要 fp16 但 CPU 又需精度 | `fp16=0` 导出（param/bin 存 fp32），半精度在运行时由 ncnn 的 `use_fp16_*` 打开 |

## 运行

```bash
./build/tinysr-webui 8014
# 浏览器打开 http://127.0.0.1:8014
```

环境变量：

| 变量 | 默认 | 说明 |
| --- | --- | --- |
| `TINYSR_MODELS` | `models` | 模型目录 |
| `TINYSR_THREADS` | 0 | CPU 线程数；0 = `hardware_concurrency()` |
| `TINYSR_OVERLAP` | 8 | tile 重叠宽度（latent 单位，1 latent = 8 px）；8 实测最优 |
| `TINYSR_FP16` | 2 | Vulkan 半精度：0 关 / 1 仅 fp16 存储 / 2 存储+运算 |
| `TINYSR_DCFIX` | 1 | 逐 tile DC 对齐：消 tile 间平均亮度漂移（banding） |
| `TINYSR_TONEFIX` | 1 | 条件偏置修正：消模型对暗部的系统性提亮 |
| `TINYSR_RESID` | 1 | **精确残差回注**（compose 后、量化前，全程 fp32，CPU +~2s）：`R = 原图 − BOX↓(输出)` 低通后加回，输出低频带精确对齐原图，SR 高频不动。tile 台阶 0.23/0.32 → **0.03/0.06** /255 |
| `TINYSR_RESID_R` | 4 | 残差低通盒式模糊半径（原图域 px）；调小会注入原图噪声，4 实测最优 |
| `TINYSR_RESID_ALPHA` | 1.0 | 残差强度（0=关闭效果） |
| `TINYSR_OCM` | 0 | Overlap Color Matching（经典 tiling 后处理，实测劣于 dcfix，保留作实验开关） |
| `TINYSR_MB` | 0 | Multiband Blending（实测与关闭无差异，保留作实验开关） |
| `TINYSR_PHASE` | 0 | 网格相位偏移（双相位推理实验用，默认关） |
| `TINYSR_WIN` | 0 | 融合窗形状（0/1 互补余弦坡等价 / 2 Hann；实测无差异） |
| `TINYSR_DUMP` | 空 | 设成目录时把各阶段中间张量落盘，便于与 Python 对拍 |

OCM / MB / PHASE / WIN 是排查过程中保留的实验开关，对照数据与结论见 [docs/research.md](docs/research.md)。

### HTTP API

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| GET | `/` | WebUI |
| GET | `/api/info` | `{"vulkan_capable":true,"scale":4,"tile":512,"step":448,"overlap":8,"threads":12}` |
| POST | `/api/sr` | multipart: `image=<file>`, `backend=vulkan\|cpu` → `image/png`，响应头 `X-Elapsed-Ms` |

```bash
curl -s -X POST http://127.0.0.1:8014/api/sr \
     -F image=@in.png -F backend=cpu -D- -o out.png | grep -i x-elapsed
```

> **关于预览画质**：4× 之后的大图（如 1950² → 7800²）动辄上亿像素。浏览器把这种图缩到
> 屏幕尺寸时走的是低质量下采样（mipmap/LOD），在天空、水面这种平滑区会显出 448 px 的
> 块状明暗差 —— 那是**显示**问题，PNG 本身没有网格缝（实测 tile 边界阶跃 ≤ 0.2/255，
> 相邻 tile 低频差 std 0.23/255）。所以前端会先在 canvas 里分级高质量缩到长边 2400 px
> 再显示，下载链接仍是原始分辨率。要看真实画质请下载 PNG 在本地 100% 查看。

## 性能参考

机器：Intel Core Ultra（Arc B370 iGPU）+ 12 线程 CPU，Linux。

| 输入 → 输出 | 后端 | tile 数 | 服务端耗时 |
| --- | --- | --- | --- |
| 512×288 → 2048×1152 | Vulkan (fp16 存储+运算) | 20 | **7.0 s** |
| 512×288 → 2048×1152 | CPU (12 线程) | 20 | 41.2 s |
| 1672×941 → 6688×3764 (25MP) | Vulkan (fp16) | 160 | **54.3 s** |
| 1672×941 → 6688×3764 (25MP) | CPU (12 线程) | 160 | 298.9 s |

tile 吞吐：Vulkan 约 0.34 s/tile，CPU 约 1.9 s/tile（约 5.5×）。

精度（Vulkan fp16 vs CPU fp32，0–255 尺度，目视无差异）：512×288 mean 0.32 / max 28；1672×941 mean 0.20 / max 55。仅 fp16 存储时误差更小（裁剪图 mean 0.11 / max 9）。默认 fp16 存储+运算，`TINYSR_FP16=1` 切保守模式，`TINYSR_FP16=0` 完全关闭半精度。

后端首次请求时才加载对应模型（约 1.4 GB 权重），切换后端会重新加载，之后常驻。

## 为什么不做 INT8

在本模型上做过完整的量化对照实验（多个后端实现结论一致）：

- **全 PTQ INT8**（MINMAX / 去 outlier / SmoothQuant 三种策略）：延迟反而慢 27%，输出与 fp32 的最大差达 2.4–2.6（0–255 尺度）——直接崩图。
- **weight-only INT8**：等速、省内存，精度可用，但收益不足以支撑在移动 GPU 上引入反量化路径。

所以发布版锁定 **fp16**：桌面/移动 GPU 都有原生 fp16 存储与 FMA，精度损失可忽略，且不需要反量化。

## 许可

- 代码（本仓库）：MIT。
- TinySR 模型/代码：遵循原仓库许可。
- SD3 底模（Locomotive / pruned backbone）：**Stability AI Community License**，商用需另行确认。

导出模型与权重不随本仓库分发（Releases 的 models.zip 或自导出）。
