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
symlinking the read-only NVFP4 text weights.

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

Add the following to your Claude Code settings file (`~/.claude/settings.json`
on Linux/macOS, `%USERPROFILE%\.claude\settings.json` on Windows).

```json
{
  "env": {
    "ANTHROPIC_BASE_URL": "http://<server's lan ip>:8000",
    "ANTHROPIC_API_KEY": "sk-local-no-key-needed",
    "ANTHROPIC_DEFAULT_SONNET_MODEL": "qwen3.8-27b",
    "ANTHROPIC_DEFAULT_OPUS_MODEL": "qwen3.8-27b",
    "ANTHROPIC_DEFAULT_HAIKU_MODEL": "qwen3.8-27b",
    "CLAUDE_CODE_SUBAGENT_MODEL": "qwen3.8-27b",
    "CLAUDE_CODE_MAX_OUTPUT_TOKENS": "120000",
    "CLAUDE_CODE_AUTO_COMPACT_WINDOW": "130000",
    "CLAUDE_CODE_ATTRIBUTION_HEADER": "0",
    "CLAUDE_CODE_USE_POWERSHELL_TOOL": "1"
  },
  "skipWebFetchPreflight": true,
  "permissions": {
    "allow": [
      "Agent",
      "Bash",
      "Edit",
      "PowerShell",
      "WebFetch",
      "Write"
    ],
    "defaultMode": "auto"
  },
  "model": "qwen3.8-27b",
  "skipDangerousModePermissionPrompt": true,
  "autoCompactEnabled": true
}
```

`CLAUDE_CODE_USE_POWERSHELL_TOOL` is only needed on Windows clients; drop it on Linux/macOS.

### Why the permissions are so permissive

`permissions.allow` pre-approves every tool this setup uses (`Bash`, `Edit`,
`Write`, `Agent`, ...), `defaultMode: auto` keeps sessions in auto mode, and
`skipDangerousModePermissionPrompt` suppresses the warning prompt. Together
they amount to unrestricted shell and file access. This is not a safe default;
use it only with a local, trusted model and a trusted working directory.

The reason is latency. In auto mode, a tool call that is not pre-approved
requires a permission decision, and that decision is another model round trip.
Against a local backend the extra round trip adds seconds to every tool call
and can time out on long-context sessions, leaving the session stalled or the
tool call dropped. Pre-approving the tool set removes that round trip and lets
tool calls run immediately. If you do not hit these timeouts, narrow
`permissions.allow` to the tools you actually need and drop
`skipDangerousModePermissionPrompt`.

The server itself binds `0.0.0.0:8000` without authentication, so anything on
the LAN can use it. Keep it on a trusted network, or change `--host 0.0.0.0`
to `--host 127.0.0.1` in the start scripts if only local access is needed.

## 5. Stop the Server

```bash
./stop.sh
```
