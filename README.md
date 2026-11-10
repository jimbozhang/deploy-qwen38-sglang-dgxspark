# Deploy Qwen3.8-27B on DGX Spark

Serve Qwen3.5-based models via [sglang](https://github.com/sgl-project/sglang) and use them as the Claude Code backend.

| Model | Script | Quantization | Speculative Decoding |
|---|---|---|---|
| [RadixArk/Qwen3.8-27B-NVFP4](https://huggingface.co/RadixArk/Qwen3.8-27B-NVFP4) | `start.sh` | NVFP4 + FP8 | [DFlash2](https://huggingface.co/z-lab/Qwen3.8-27B-DFlash2) |
| [DavidAU Qwen3.8-27B-TURBO-Fable-...-Heretic](https://huggingface.co/DavidAU/Qwen3.8-27B-TURBO-Fable-Cold-Fusion-735-882-Heretic-Uncensored-NM-DAU) | `start_heretic.sh` | NVFP4 + vision tower (from the original BF16 checkpoint) | None |

## 1. Download Models

```bash
export MODEL_ROOT=~/pretrained

pip install huggingface_hub

# Base model
huggingface-cli download RadixArk/Qwen3.8-27B-NVFP4 \
  --local-dir $MODEL_ROOT/RadixArk/Qwen3.8-27B-NVFP4

# DFlash2 draft model
huggingface-cli download z-lab/Qwen3.8-27B-DFlash2 \
  --local-dir $MODEL_ROOT/z-lab/Qwen3.8-27B-DFlash2

# Heretic model — by DavidAU, NVFP4 quantized by Solstice-AI (text-only)
huggingface-cli download Solstice-AI/Qwen3.8-27B-TURBO-Fable-Cold-Fusion-735-882-Heretic-Uncensored-NM-DAU-NVFP4 \
  --local-dir $MODEL_ROOT/Solstice-AI/Qwen3.8-27B-TURBO-Fable-Cold-Fusion-735-882-Heretic-Uncensored-NM-DAU-NVFP4

# Heretic vision source — full BF16 original (contains the vision tower)
huggingface-cli download DavidAU/Qwen3.8-27B-TURBO-Fable-Cold-Fusion-735-882-Heretic-Uncensored-NM-DAU \
  --local-dir $MODEL_ROOT/DavidAU/Qwen3.8-27B-TURBO-Fable-Cold-Fusion-735-882-Heretic-Uncensored-NM-DAU

# If huggingface.co is blocked, use the mirror:
#   export HF_ENDPOINT=https://hf-mirror.com
```

The Heretic NVFP4 quant dropped the vision tower. `setup_heretic_vl.sh` builds
a vision-enabled overlay under `models/heretic-735-882-vl/` (gitignored) by
extracting the 333 `model.visual.*` tensors from the original checkpoint and
symlinking the read-only NVFP4 text weights. `~/pretrained` is never modified.

```bash
./setup_heretic_vl.sh
```

## 2. Pull Docker Image

```bash
docker pull lmsysorg/sglang:dev-qwen38-27b-dflash2
```

## 3. Start the Server

```bash
# Option A: RadixArk + DFlash2 (default, fastest)
./start.sh

# Option B: DavidAU Heretic (uncensored, vision-enabled, no DFlash2)
./start_heretic.sh
```

Wait until the log shows `The server is fired up and ready to roll!`, then verify:

```bash
curl http://localhost:8000/v1/models
```

## 4. Configure Claude Code

Copy `settings.json` to your Claude Code settings, replacing `192.168.31.51` with your server's LAN IP.

## 5. Stop the Server

```bash
./stop.sh
```
