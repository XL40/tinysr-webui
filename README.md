# TinySR WebUI

**English** · [简体中文](README.zh-CN.md)

One-click upscaler for **[TinySR](https://github.com/Microtreei/TinySR)** (CVPR 2026 Findings) — a pruned single-step diffusion transformer (DiT) super-resolution model — running entirely on **ncnn** with **fp16**, on **Vulkan GPU** or **CPU**. No PyTorch / diffusers / OpenVINO at runtime; a single command serves a web page that upscales any image 4×.

## Features

- **Single stack**: C++ + ncnn, no Python runtime dependency (PyTorch only needed when exporting models).
- **Dual backend**: switch between Vulkan GPU (fp16 storage + compute) and CPU (fp32) in the web UI.
- **Any resolution**: 64-latent tile grid with overlap-region complementary cosine blending; input size is not restricted.
- **Seamless output**: per-tile DC alignment + exposure calibration + exact residual reinjection — no visible block seams in flat/dark regions.
- **fp16, not INT8**: fp16 is mature on mobile/desktop GPUs and needs no dequantization path; INT8 quantization does not pay off for this model (see [Why no INT8](#why-no-int8)).
- **Self-contained frontend**: a single `index.html` — drag & drop, before/after slider with zoom/pan, mix-strength control, format selection, download.
- **Image formats**: input WebP / TIFF (8/16-bit) / AVIF (SDR + HDR) / DNG; output PNG / JPEG / WebP / AVIF / BMP. HDR (PQ/HLG) passes through at the source bit depth.

## Architecture

```
Browser (drag & drop / compare slider)
   │  multipart POST /api/sr  (image, backend=vulkan|cpu, scale, fmt)
   ▼
cpp-httplib  (src/main.cpp)
   ▼
Engine::upscale_f  (src/engine.cpp)
   ├─ bicubic upscale → [-1,1] → pad to multiples of 512
   ├─ vae_encoder   : 512px window → 16×64×64 latent   ┐
   ├─ transformer   : 16×64×64 → 16×64×64 (z−pred)      ├ shared latent tile grid,
   ├─ vae_decoder   : 16×64×64 → 512px window           ┘ cosine blending on overlaps
   ├─ per-tile DC alignment + tonefix (kills inter-tile luma drift / shadow lift)
   ├─ exact residual reinjection: output low band aligned to source, all fp32
   └─ encode to png / jpg / webp / avif / bmp (X-Elapsed-Ms response header)
```

Hard constraint on the model side: `pos_embed` is fixed at 1024 tokens (32×32 patch grid = 512px tile = 64×64 latent), so arbitrary resolutions must be tiled.

Root-cause analysis, alternatives and measurements for tiling consistency: **[docs/research.md](docs/research.md)**.

## Prebuilt packages (Releases)

Download the zip for your platform from [Releases](../../releases) (binary + frontend), plus `models.zip` unpacked next to the binary:

```
tinysr-webui.exe (or tinysr-webui)
web/index.html
models/*.ncnn.param / *.ncnn.bin   ← from models.zip
```

A Vulkan loader from the GPU driver is required at runtime (Windows: `vulkan-1.dll`, Linux: `libvulkan.so.1` — present once NVIDIA/AMD/Intel drivers are installed). CPU-only machines: build from source with `-DTINYSR_VULKAN=OFF`.

> Windows release builds ship with stb built-in formats only (PNG/JPEG/BMP in/out); the optional system libraries (WebP/TIFF/AVIF/DNG) require a vcpkg-based toolchain on Windows CI and are not wired up yet.

## Build

Requirements: CMake ≥ 3.14, a C++17 compiler, Vulkan loader (`libvulkan.so`) for the GPU backend.

```bash
# 1. Fetch dependencies (ncnn source + glslang submodule + 4 single-header libs)
./tools/setup_third_party.sh
#    Behind a proxy: HTTPS_PROXY=http://127.0.0.1:PORT ./tools/setup_third_party.sh

# 2. Build (first build compiles ncnn + glslang, ~3-5 min on 12 cores)
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build
```

CPU-only build: `cmake -B build -DTINYSR_VULKAN=OFF` (the frontend auto-detects and disables the GPU option).

### Image formats (optional system libraries)

Only royalty-free / open-source formats are supported. **HEIC/HEIF is not supported** (HEVC patent pools are royalty-bearing; use AVIF for the same purpose). Each format is probed at configure time — if the library is missing, that format is silently disabled and the build still succeeds.

| Format | Input | Output | Dependency (auto-detected) |
| --- | --- | --- | --- |
| JPEG / PNG / BMP / GIF (first frame) / TGA / PNM / PSD | ✓ | JPEG / PNG / BMP | built-in stb (none) |
| WebP | ✓ | ✓ (q90) | libwebp |
| TIFF (grayscale/RGB, 8/16-bit) | ✓ | — | libtiff |
| AVIF (SDR 8-bit; **HDR 10/12-bit PQ/HLG keeps source bit depth**) | ✓ | ✓ (SDR 8-bit q90, HDR same depth; SVT-AV1/aom encoder, slower on large images) | libavif |
| DNG (CFA demosaic + camera WB, decoded to sRGB 16-bit) | ✓ | — | LibRaw |

Debian/Ubuntu: `sudo apt install libwebp-dev libtiff-dev libavif-dev libraw-dev` · Fedora: `sudo dnf install libwebp-devel libtiff-devel libavif-devel LibRaw-devel`.

HDR pipeline: PQ/HLG sources are inferred in the perceptual code-domain end to end; AVIF output preserves the source bit depth and NCLX color properties. The frontend disables client-side mixing for HDR/high-bit-depth results (canvas is 8-bit only).

The **mix strength** dropdown (100% default, 10% steps) blends the SR result with an upscaled interpolation anchor entirely in the browser — per pixel `out = m·SR + (1−m)·anchor`. Adjusting it re-blends from the cached result and **never re-runs inference**.

Dependency versions are pinned in `tools/setup_third_party.sh` (ncnn `71f7318` / cpp-httplib `v0.66.0` / stb `9600522` / stb_image_resize2 `a4f303e`), with `HTTPS_PROXY` support and per-file size/version verification.

### Windows

Same toolchain (`cmake -B build` + `cmake --build build --config Release` in PowerShell):

- Vulkan SDK required at build time (links `vulkan-1.lib`): [LunarG](https://vulkan.lunarg.com/sdk/home) or `choco install vulkan-sdk`
- MSVC: CMake adds `/utf-8`, `NOMINMAX`, and static-links the runtime (no VC Redist needed on user machines)

### CI / Release

- **ci.yml**: Ubuntu + Windows matrix build + smoke test (starts the server, checks `/api/info`) on push/PR.
- **release.yml**: triggered by `v*` tags; packages both platforms and attaches them to the GitHub Release. Model weights never enter CI — upload `models.zip` manually with `gh release upload <tag> models.zip`.

## Models

The 6 files under `models/` (`*.ncnn.param` / `*.ncnn.bin`) must be exported yourself; weights (1.3 GB) are not in the repo:

```bash
python tools/export_ncnn.py \
  --repo  /path/to/TinySR-main \
  --ckpt  /path/to/checkpoint \
  --out   models
```

The script merges LoRA → `torch.jit.trace(check_trace=False)` → pnnx → ncnn. Verified against torch fp32: max abs diff ≤ 1.5e-5 across all three models.

Pitfalls already handled in the script:

| Problem | Handling |
| --- | --- |
| `torch.jit.trace` default `check_trace=True` false-positives on this graph | `check_trace=False`; correctness is verified by an independent post-export comparison |
| pnnx cannot convert 6D `reshape/permute` (unpatchify) | unpatchify rewritten as a fixed-weight `ConvTranspose2d(k=2,s=2)` (PixelShuffle-equivalent), 4D only |
| `ConvTranspose2d` weight layout is easy to get wrong | shape is `[in, out, kh, kw]` (opposite of `Conv2d`) |
| fp16 wanted, but CPU needs precision | export with `fp16=0` (fp32 param/bin); enable half precision at runtime via ncnn `use_fp16_*` |

## Run

```bash
./build/tinysr-webui 8014 [--verbose]
# open http://127.0.0.1:8014
```

Environment variables:

| Variable | Default | Description |
| --- | --- | --- |
| `TINYSR_MODELS` | `models` | model directory |
| `TINYSR_SCALE` | 4 | default upscale factor (1–8); used when the UI preset and request `scale=` are unset |
| `TINYSR_VERBOSE` | 0 | 1 = per-tile progress on stderr (same as `--verbose`) |
| `TINYSR_THREADS` | 0 | CPU threads; 0 = `hardware_concurrency()` |
| `TINYSR_OVERLAP` | 8 | tile overlap (latent units, 1 latent = 8 px) |
| `TINYSR_FP16` | 2 | Vulkan half precision: 0 off / 1 fp16 storage / 2 storage + compute |
| `TINYSR_DCFIX` | 1 | per-tile DC alignment (kills inter-tile luma drift / banding) |
| `TINYSR_TONEFIX` | 1 | conditional bias correction (kills systematic shadow lift) |
| `TINYSR_FLATGATE` | 1 | **detail budget gating**: defocus-evidence detection on the LR image; evidenced regions keep model detail, evidence-free flat areas get Laplacian band suppression (λ=0.6/0.9) plus a detail budget clamp of `4×activity + 1.5/255`. 0 = off |
| `TINYSR_RESID` | 1 | **exact residual reinjection** (post-compose, pre-quantize, all fp32, CPU +~2s): `R = source − BOX↓(output)`, low-passed and added back — the output low band exactly matches the source, SR high frequencies untouched |
| `TINYSR_RESID_R` | 4 | residual low-pass box radius (source-domain px) |
| `TINYSR_RESID_ALPHA` | 1.0 | residual strength |
| `TINYSR_DUMP` | — | set to a directory to dump intermediate tensors of each stage (for diffing against Python) |

### HTTP API

| Method | Path | Description |
| --- | --- | --- |
| GET | `/` | WebUI (zoom/pan compare viewer, wheel zoom, drag pan, freely draggable split line) |
| GET | `/api/info` | `{"vulkan_capable":true,"gpu":"…","scale":4,"tile":512,"step":448,"overlap":8,"threads":12,"webp":true,"avif":true,"dng":true,…}` |
| GET | `/api/progress` | current job progress `{job,stage,done,total,elapsed_ms}` (polled by the frontend while uploading) |
| POST | `/api/sr` | multipart: `image=<file>`, `backend=vulkan\|cpu`, `scale=2\|3\|4…` (default `TINYSR_SCALE`), `fmt=png\|jpg\|bmp\|webp\|avif` (default png) → `image/*` with `X-Elapsed-Ms` / `X-Bit-Depth` / `X-Hdr` headers |
| POST | `/api/probe` | multipart: `image=<file>` → `{"w","h","bits","hdr","fmt"}` (for formats the browser cannot decode, e.g. DNG) |

```bash
curl -s -X POST http://127.0.0.1:8014/api/sr \
     -F image=@in.png -F backend=cpu -D- -o out.png | grep -i x-elapsed
```

## Performance

Machine: Intel Core Ultra (Arc B370 iGPU) + 12-thread CPU, Linux.

| Input → Output | Backend | Tiles | Server time |
| --- | --- | --- | --- |
| 512×288 → 2048×1152 | Vulkan (fp16 storage+compute) | 20 | **7.0 s** |
| 512×288 → 2048×1152 | CPU (12 threads) | 20 | 41.2 s |
| 1672×941 → 6688×3764 (25MP) | Vulkan (fp16) | 160 | **54.3 s** |
| 1672×941 → 6688×3764 (25MP) | CPU (12 threads) | 160 | 298.9 s |

Throughput: ~0.34 s/tile on Vulkan, ~1.9 s/tile on CPU (~5.5×).

The 2× mode (`scale=2`, selectable in the UI) covers 1/4 the tiles of 4× for the same input: ~3–4.4× faster; round-trip fidelity PSNR 33.4 vs 36.2 (night) / 39.6 vs 42.5 (bright) — an explicit speed-for-fidelity trade-off (smaller bicubic pre-upscale means each 512px tile covers twice the source area).

Precision (Vulkan fp16 vs CPU fp32, 0–255 scale, visually identical): 512×288 mean 0.32 / max 28; 1672×941 mean 0.20 / max 55. Storage-only fp16 is tighter (mean 0.11 / max 9). Default is storage+compute; `TINYSR_FP16=1` for the conservative mode, `TINYSR_FP16=0` to disable.

Backends load their model (≈1.4 GB weights) on first request and stay resident; switching backends reloads.

## Why no INT8

Full quantization experiments were run on this model (consistent across backends):

- **Full PTQ INT8** (MINMAX / outlier removal / SmoothQuant): 27% *slower*, and output deviates from fp32 by up to 2.4–2.6 (0–255) — broken results.
- **Weight-only INT8**: same speed, saves memory, accuracy acceptable — but not worth adding a dequantization path on mobile GPUs.

Releases therefore ship **fp16** only: native fp16 storage and FMA on desktop/mobile GPUs, negligible accuracy loss, no dequantization.

## License

- Code (this repo): MIT.
- TinySR model/code: follow the upstream repository license.
- SD3 base model (Locomotive / pruned backbone): **Stability AI Community License** — check terms before commercial use.

Exported models and weights are not distributed with this repo (grab `models.zip` from Releases or export your own).
