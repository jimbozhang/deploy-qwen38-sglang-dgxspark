#!/bin/bash

MODEL_ROOT="$HOME/pretrained"
CACHE_DIR="$HOME/.cache/qwen38/sglang"

# Patch chat template for Claude Code (effort levels + tool_reference blocks)
PATCHED_TEMPLATE="$CACHE_DIR/chat_template_patched.jinja"
bash "$(dirname "$0")/patch_chat_template.sh" \
  "$MODEL_ROOT/RadixArk/Qwen3.8-27B-NVFP4/chat_template.jinja" \
  "$PATCHED_TEMPLATE"

docker run -d --name qwen38-sglang \
  --gpus all --ipc=host --shm-size=16g \
  -p 8000:8000 \
  -e NVIDIA_VISIBLE_DEVICES=all \
  -e NVIDIA_DRIVER_CAPABILITIES=compute,utility \
  -e HF_HUB_DISABLE_XET=1 \
  -e HF_HUB_OFFLINE=1 \
  -e TRANSFORMERS_OFFLINE=1 \
  -e HF_HOME=/root/.cache/huggingface \
  -e TORCHINDUCTOR_CACHE_DIR=/cache/inductor \
  -v "$MODEL_ROOT:/models:ro" \
  -v "$HOME/.cache/qwen38/hf:/root/.cache/huggingface" \
  -v "$CACHE_DIR:/cache" \
  lmsysorg/sglang:dev-qwen38-27b-dflash2 \
  python3 -m sglang.launch_server \
    --trust-remote-code \
    --model-path /models/RadixArk/Qwen3.8-27B-NVFP4 \
    --chat-template /cache/chat_template_patched.jinja \
    --served-model-name qwen3.8-27b \
    --tp-size 1 \
    --host 0.0.0.0 --port 8000 \
    --context-length 262144 \
    --attention-backend flashinfer \
    --kv-cache-dtype fp8_e4m3 \
    --chunked-prefill-size 8192 \
    --disable-prefill-cuda-graph \
    --cuda-graph-max-bs 8 \
    --disable-flashinfer-autotune \
    --mem-fraction-static 0.60 \
    --reasoning-parser qwen3 \
    --tool-call-parser qwen3_coder \
    --speculative-algorithm DFLASH \
    --speculative-draft-model-path /models/z-lab/Qwen3.8-27B-DFlash2 \
    --speculative-num-draft-tokens 8 \
    --speculative-draft-model-quantization unquant \
    --mamba-radix-cache-strategy extra_buffer \
    --mamba-ssm-dtype bfloat16 \
    --max-mamba-cache-size 144 \
    --max-running-requests 6 \
    --allow-auto-truncate \
    --enable-metrics

echo "Following logs (Ctrl+C to stop, container keeps running)..."
docker logs -f qwen38-sglang
