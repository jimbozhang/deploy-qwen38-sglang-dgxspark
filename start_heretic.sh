#!/bin/bash
# Start sglang with the DavidAU Heretic uncensored NVFP4 model, with vision.
# Original model by DavidAU, NVFP4 quantization by Solstice-AI.
# The vision tower is grafted from the original BF16 checkpoint into an
# overlay under models/ (see setup_heretic_vl.sh); ~/pretrained stays read-only.
# No DFlash2 speculative decoding.

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
MODEL_ROOT="$HOME/pretrained"
CACHE_DIR="$HOME/.cache/qwen38/sglang"
MODEL_PATH="Solstice-AI/Qwen3.8-27B-TURBO-Fable-Cold-Fusion-735-882-Heretic-Uncensored-NM-DAU-NVFP4"
OVERLAY_DIR="$SCRIPT_DIR/models/heretic-735-882-vl"

MODEL_DIR="$MODEL_ROOT/$MODEL_PATH"
if [ ! -d "$MODEL_DIR" ]; then
  echo "Error: Model directory not found: $MODEL_DIR"
  echo "Download the model first:"
  echo "  huggingface-cli download $MODEL_PATH --local-dir $MODEL_DIR"
  exit 1
fi

if [ ! -d "$OVERLAY_DIR" ]; then
  echo "Error: Vision overlay not found: $OVERLAY_DIR"
  echo "Build it first (downloads nothing, reads ~/pretrained):"
  echo "  ./setup_heretic_vl.sh"
  exit 1
fi

# Patch chat template for Claude Code (effort levels + tool_reference blocks)
PATCHED_TEMPLATE="$CACHE_DIR/chat_template_patched_heretic.jinja"
SOURCE_TEMPLATE="$OVERLAY_DIR/chat_template.jinja"
if [ ! -f "$SOURCE_TEMPLATE" ]; then
  echo "Warning: No chat_template.jinja found, using Solstice-AI template"
  SOURCE_TEMPLATE="$MODEL_DIR/chat_template.jinja"
fi
bash "$SCRIPT_DIR/patch_chat_template.sh" "$SOURCE_TEMPLATE" "$PATCHED_TEMPLATE"

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
  -v "$MODEL_ROOT:/pretrained:ro" \
  -v "$OVERLAY_DIR:/overlay:ro" \
  -v "$HOME/.cache/qwen38/hf:/root/.cache/huggingface" \
  -v "$CACHE_DIR:/cache" \
  lmsysorg/sglang:dev-qwen38-27b-dflash2 \
  python3 -m sglang.launch_server \
    --trust-remote-code \
    --model-path /overlay \
    --chat-template "/cache/chat_template_patched_heretic.jinja" \
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
    --mamba-radix-cache-strategy extra_buffer \
    --mamba-ssm-dtype bfloat16 \
    --max-mamba-cache-size 144 \
    --max-running-requests 6 \
    --allow-auto-truncate \
    --enable-metrics

echo "Following logs (Ctrl+C to stop, container keeps running)..."
echo "Model: $MODEL_PATH"
docker logs -f qwen38-sglang
