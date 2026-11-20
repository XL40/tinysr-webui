# models/

这里放导出好的 ncnn 模型（不入库，1.4 GB）：

```
tinysr_transformer.ncnn.param / .bin    # MMDiT (LoRA 已合并), 355M 参数
vae_encoder.ncnn.param       / .bin
vae_decoder.ncnn.param       / .bin
```

生成方式见仓库根目录 README 的「模型」一节：

```bash
python tools/export_ncnn.py --repo /path/to/TinySR-main --ckpt /path/to/checkpoint --out models
```
