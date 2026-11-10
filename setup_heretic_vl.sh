#!/bin/bash
# Build a vision-capable overlay for the Solstice NVFP4 Heretic model without
# touching ~/pretrained.
#
# The Solstice NVFP4 quant is text-only: it dropped the vision tower. The
# original DavidAU 735-882 checkpoint (full BF16) has the matching tower in
# model-00012-of-00012.safetensors. This script:
#   1. extracts the 333 model.visual.* tensors to models/heretic-735-882-vl/
#   2. symlinks the read-only NVFP4 text weights/tokenizer into the overlay
#   3. writes a patched config.json and model.safetensors.index.json
#
# Usage: ./setup_heretic_vl.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
MODEL_ROOT="${MODEL_ROOT:-$HOME/pretrained}"
SOLSTICE_DIR="$MODEL_ROOT/Solstice-AI/Qwen3.8-27B-TURBO-Fable-Cold-Fusion-735-882-Heretic-Uncensored-NM-DAU-NVFP4"
ORIGINAL_DIR="$MODEL_ROOT/DavidAU/Qwen3.8-27B-TURBO-Fable-Cold-Fusion-735-882-Heretic-Uncensored-NM-DAU"
OVERLAY_DIR="$SCRIPT_DIR/models/heretic-735-882-vl"
IMAGE="lmsysorg/sglang:dev-qwen38-27b-dflash2"

for d in "$SOLSTICE_DIR" "$ORIGINAL_DIR"; do
  if [ ! -d "$d" ]; then
    echo "Error: missing model directory: $d"
    exit 1
  fi
done

mkdir -p "$OVERLAY_DIR"

echo "Extracting vision tower from the original checkpoint..."
docker run --rm -i \
  -v "$MODEL_ROOT:/pretrained:ro" \
  -v "$OVERLAY_DIR:/out" \
  --entrypoint python3 \
  "$IMAGE" - <<'PY'
import json
import os

from safetensors import safe_open
from safetensors.torch import save_file

orig = "/pretrained/DavidAU/Qwen3.8-27B-TURBO-Fable-Cold-Fusion-735-882-Heretic-Uncensored-NM-DAU"
out = "/out/vision.safetensors"

index = json.load(open(f"{orig}/model.safetensors.index.json"))["weight_map"]
visual = {k: v for k, v in index.items() if k.startswith("model.visual.")}
assert len(visual) == 333, f"expected 333 visual tensors, found {len(visual)}"

by_shard = {}
for key, shard in visual.items():
    by_shard.setdefault(shard, []).append(key)

tensors = {}
for shard, keys in by_shard.items():
    with safe_open(f"{orig}/{shard}", framework="pt") as f:
        for key in keys:
            tensors[key] = f.get_tensor(key).contiguous().cpu()

save_file(tensors, out)
total = sum(t.numel() * t.element_size() for t in tensors.values())
print(f"  extracted {len(tensors)} tensors, {total} bytes -> {out}")
PY

# The extraction container runs as root; make the file readable by the host user.
docker run --rm -v "$OVERLAY_DIR:/out" --entrypoint chmod "$IMAGE" a+r /out/vision.safetensors

echo "Symlinking read-only NVFP4 files..."
solstice_name="$(basename "$SOLSTICE_DIR")"
for f in model.safetensors model-mtp-restored.safetensors tokenizer.json \
         tokenizer_config.json generation_config.json preprocessor_config.json \
         processor_config.json video_preprocessor_config.json recipe.yaml; do
  if [ -e "$SOLSTICE_DIR/$f" ]; then
    ln -sfn "../pretrained/Solstice-AI/$solstice_name/$f" "$OVERLAY_DIR/$f"
  fi
done

# Use the original model's own template (8.9KB stock Qwen3.5 template). The
# 24.8KB template shipped in the NVFP4 repo is a different custom template.
cp "$ORIGINAL_DIR/chat_template.jinja" "$OVERLAY_DIR/chat_template.jinja"

echo "Writing patched config.json and index..."
SOLSTICE_DIR="$SOLSTICE_DIR" ORIGINAL_DIR="$ORIGINAL_DIR" OVERLAY_DIR="$OVERLAY_DIR" python3 - <<'PY'
import json
import os
import struct

solstice = os.environ["SOLSTICE_DIR"]
original = os.environ["ORIGINAL_DIR"]
overlay = os.environ["OVERLAY_DIR"]

# --- config.json: add the vision fields, enable the visual module ---
cfg = json.load(open(f"{solstice}/config.json"))
orig_cfg = json.load(open(f"{original}/config.json"))
for key in (
    "vision_config",
    "image_token_id",
    "video_token_id",
    "vision_start_token_id",
    "vision_end_token_id",
):
    cfg[key] = orig_cfg[key]
cfg["language_model_only"] = False
json.dump(cfg, open(f"{overlay}/config.json", "w"), indent=2)

# --- index: point the 333 visual keys at vision.safetensors ---
idx = json.load(open(f"{solstice}/model.safetensors.index.json"))
orig_idx = json.load(open(f"{original}/model.safetensors.index.json"))
visual = {
    k: v
    for k, v in orig_idx["weight_map"].items()
    if k.startswith("model.visual.")
}
assert len(visual) == 333, f"expected 333 visual keys, found {len(visual)}"
idx["weight_map"].update({k: "vision.safetensors" for k in visual})

def data_bytes(path):
    with open(path, "rb") as f:
        header_len = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(header_len))
    return sum(
        v["data_offsets"][1] - v["data_offsets"][0]
        for k, v in header.items()
        if k != "__metadata__"
    )

vision_bytes = data_bytes(f"{overlay}/vision.safetensors")
text_bytes = data_bytes(f"{solstice}/model.safetensors") + data_bytes(
    f"{solstice}/model-mtp-restored.safetensors"
)

meta = idx.setdefault("metadata", {})
meta["total_size"] = text_bytes + vision_bytes
meta["total_tensors"] = len(idx["weight_map"])
meta["total_shards"] = meta.get("total_shards", 2) + 1
json.dump(idx, open(f"{overlay}/model.safetensors.index.json", "w"), indent=2)

print(f"  config: language_model_only=False, vision tensors={len(visual)}")
print(f"  index: {len(idx['weight_map'])} keys, +{vision_bytes} bytes")
PY

echo
echo "Overlay ready: $OVERLAY_DIR"
ls -l "$OVERLAY_DIR"
