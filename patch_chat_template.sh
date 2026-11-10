#!/bin/bash
# Patch a Qwen3.8 chat template for use behind Claude Code.
#
# 1. Map Claude Code's 'high'/'max' effort levels onto the levels the
#    template understands. No-op for templates without reasoning_effort.
# 2. Ignore tool_reference content blocks instead of raising. Claude Code
#    emits these when ToolSearch loads a deferred tool. The referenced
#    schemas are already rendered in the <tools> block, so dropping the
#    reference is safe.
#
# Usage: patch_chat_template.sh <source_template> <destination_template>

set -e

SRC="$1"
DST="$2"

if [ -z "$SRC" ] || [ -z "$DST" ]; then
  echo "Usage: $0 <source_template> <destination_template>"
  exit 1
fi

mkdir -p "$(dirname "$DST")"
cp "$SRC" "$DST"

# Map 'high' -> 'medium' and 'max' -> 'xhigh'. GNU sed turns each \n escape
# into a line break; templates without the anchor are left unchanged.
sed -i "/resolved_reasoning_effort = reasoning_effort|default/a\\    {%- if resolved_reasoning_effort == 'high' %}\\n        {%- set resolved_reasoning_effort = 'medium' %}\\n    {%- elif resolved_reasoning_effort == 'max' %}\\n        {%- set resolved_reasoning_effort = 'xhigh' %}\\n    {%- endif %}" "$DST"

# Claude Code sends tool_reference blocks inside tool_result content after a
# ToolSearch call. The templates only understand image/video/text items and
# otherwise raise, which turns every later request in that session into a 500.
sed -i "s#raise_exception('Unexpected item type in content.')#('' if item.type is defined and item.type == 'tool_reference' else raise_exception('Unexpected item type in content.'))#" "$DST"

echo "Patched chat template: $DST"
