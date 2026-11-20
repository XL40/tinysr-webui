#!/usr/bin/env python3
"""TinySR -> ncnn 导出链 (WebUI 版, ncnn 单栈 + fp16)

torch(fp32, LoRA merged) -> torchscript -> pnnx -> ncnn param/bin

用法:
    python tools/export_ncnn.py --repo /path/to/TinySR-main \
                                --ckpt /path/to/checkpoint \
                                --out  models

产出 (WebUI 直接读取):
    models/tinysr_transformer.ncnn.{param,bin}
    models/vae_encoder.ncnn.{param,bin}
    models/vae_decoder.ncnn.{param,bin}

要点 / 已知坑:
  * torch.jit.trace 必须 check_trace=False, deepcopy 自校验对该图误报。
  * pnnx 无法转换 6D reshape/permute -> unpatchify 用固定权重
    ConvTranspose2d(k=2,s=2) 表达 (PixelShuffle 等价), 绕开 6D。
  * ConvTranspose2d weight 形状是 [in, out, kh, kw] (与 Conv 相反)。
  * fp16=0 导出: param/bin 存 fp32, 半精度由运行时 ncnn opt 控制
    (Vulkan fp16 storage/arithmetic), 这样 CPU 后端仍走 fp32 精确路径。
"""
import argparse
import os
import pathlib
import subprocess
import sys

import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=os.environ.get("TINYSR_REPO", ""),
                    help="TinySR 源码仓库 (含 models/)")
    ap.add_argument("--ckpt", default=os.environ.get("TINYSR_CKPT", ""),
                    help="checkpoint 目录 (含 tinybackbone/ vae/ tinysr/)")
    ap.add_argument("--out", default="models", help="输出目录")
    ap.add_argument("--lora-rank", type=int, default=64)
    ap.add_argument("--pnnx", default=None, help="pnnx 可执行文件路径")
    args = ap.parse_args()

    repo = pathlib.Path(args.repo).expanduser().resolve()
    ckpt = pathlib.Path(args.ckpt).expanduser().resolve()
    out = pathlib.Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    pnnx = args.pnnx or "pnnx"

    if not (repo / "models").is_dir():
        sys.exit(f"--repo 无效: {repo}")
    if not (ckpt / "tinysr").is_dir():
        sys.exit(f"--ckpt 无效: {ckpt}")

    sys.path.insert(0, str(repo))
    os.chdir(str(repo))
    _orig = torch.load
    torch.load = lambda *a, **k: _orig(*a, **{**k, "map_location": "cpu"})

    from models.tinysr.tinysd3 import TinySD3Transformer2DModel
    from models.vae.autoencoder_tiny import AutoencoderTiny
    from peft import LoraConfig
    from diffusers import StableDiffusion3Pipeline
    from utils.util import load_lora_state_dict

    LATENT, PX = 64, 512

    print("[1] load transformer + merge LoRA ...")
    t = TinySD3Transformer2DModel.from_pretrained(
        ckpt / "tinybackbone/prune-12-merge-tinysr", subfolder="transformer",
        torch_dtype=torch.float32, low_cpu_mem_usage=False,
        ignore_mismatched_sizes=True)
    t.add_adapter(LoraConfig(
        r=args.lora_rank, lora_alpha=args.lora_rank, init_lora_weights="gaussian",
        target_modules=["to_k", "to_q", "to_v", "to_out.0", "proj",
                        "linear", "linear_1", "linear_2", "net.2"]))
    sd = StableDiffusion3Pipeline.lora_state_dict(
        str(ckpt / "tinysr"), weight_name="transformer.safetensors")
    load_lora_state_dict(sd, t)
    t.merge_and_unload()
    vae = AutoencoderTiny.from_pretrained(ckpt / "vae/separable",
                                          torch_dtype=torch.float32).eval()
    pooled = torch.load("dataset/default/pool_embeds.pt", map_location="cpu").float()
    if pooled.dim() == 1:
        pooled = pooled.unsqueeze(0)

    class TinySRCore(torch.nn.Module):
        """z -> z - pred (与 NPU 版 / engine.cpp 语义一致)"""

        def __init__(self, m, pooled):
            super().__init__()
            self.t = m
            self.register_buffer("timestep", torch.tensor([1000.0]))
            self.register_buffer("pooled", pooled)
            self.gh = self.gw = LATENT // m.config.patch_size
            self.p = m.config.patch_size
            self.ch = m.out_channels
            p, ch = self.p, self.ch
            # unpatchify 的 PixelShuffle 等价形式:
            # out[c, 2y+q, 2x+r] = w[k, c, q, r] * in[k, y, x]
            # k = q*(p*ch) + r*ch + c  (proj_out 通道内存序 q,r,c)
            w = torch.zeros(p * p * ch, ch, p, p)
            for c in range(ch):
                for q in range(p):
                    for r in range(p):
                        w[q * (p * ch) + r * ch + c, c, q, r] = 1.0
            self.unpatch = torch.nn.ConvTranspose2d(
                p * p * ch, ch, kernel_size=p, stride=p, bias=False)
            self.unpatch.weight.data = w
            self.unpatch.weight.requires_grad_(False)

        def forward(self, z):
            m = self.t
            h = m.pos_embed(z)
            temb = m.time_text_embed(self.timestep, self.pooled)
            for blk in m.transformer_blocks:
                h = blk(hidden_states=h, temb=temb)
            h, _, _ = m.norm_out(h, temb)
            h = m.proj_out(h)                      # [1, 1024, 64]
            h = h.permute(0, 2, 1).view(1, self.p * self.p * self.ch,
                                        self.gh, self.gw)
            h = self.unpatch(h)                    # [1, 16, 64, 64]
            return z - h

    def export(name, module, shape):
        print(f"[export {name}] trace -> pnnx -> ncnn ...")
        param = out / f"{name}.ncnn.param"
        binp = out / f"{name}.ncnn.bin"
        ts = str(out / f"{name}.pt")
        dummy = torch.randn(*shape)
        with torch.no_grad():
            torch.jit.trace(module, dummy, check_trace=False).save(ts)
        r = subprocess.run(
            [pnnx, ts, f"inputshape=[{','.join(map(str, shape))}]",
             f"pnnxparam={out}/{name}.pnnx.param",
             f"pnnxbin={out}/{name}.pnnx.bin",
             f"pnnxpy={out}/{name}_pnnx.py",
             f"pnnxonnx={out}/{name}.pnnx.onnx",
             f"ncnnparam={param}", f"ncnnbin={binp}",
             f"ncnnpy={out}/{name}_ncnn.py", "fp16=0"],
            capture_output=True, text=True, cwd=str(out))
        if r.returncode != 0:
            print(r.stdout[-2500:])
            print(r.stderr[-2500:])
            raise RuntimeError(f"pnnx failed for {name}")
        for f in (param, binp):
            assert f.exists(), f"missing {f}"
            print(f"    {f.name}  {f.stat().st_size/1e6:.1f}MB")
        # 中间产物 (torchscript / pnnx 各格式) 不留, 只保留 ncnn param+bin
        for pat in (f"{name}.pt", f"{name}.pnnx.param", f"{name}.pnnx.bin",
                    f"{name}.pnnx.onnx", f"{name}_pnnx.py", f"{name}_ncnn.py"):
            p = out / pat
            if p.exists():
                p.unlink()

    print("[2] build wrappers ...")
    core = TinySRCore(t, pooled).eval()
    export("tinysr_transformer", core, (1, 16, LATENT, LATENT))
    export("vae_encoder", vae.encoder.eval(), (1, 3, PX, PX))
    export("vae_decoder", vae.decoder.eval(), (1, 16, LATENT, LATENT))
    print(f"DONE -> {out}")


if __name__ == "__main__":
    main()
