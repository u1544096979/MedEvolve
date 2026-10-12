#!/usr/bin/env bash
# Usage: ./render_appendix_figures.sh  (needs node, chromium, Noto Sans SC + Roboto Mono installed)
set -euo pipefail
cd "$(dirname "$0")"
node make_appendix_figures.mjs
for n in appendix-prompts-overview appendix-verification-depth appendix-case-flow; do
  [ -f "$n.svg" ] || continue
  w=$(grep -o 'width="[0-9]*"' "$n.svg" | head -1 | grep -o '[0-9]*'); h=$(grep -o 'height="[0-9]*"' "$n.svg" | head -1 | grep -o '[0-9]*')
  html="/tmp/$n.html"
  printf '<!doctype html><meta charset=utf-8><style>@page{size:%spx %spx;margin:0}html,body{margin:0}svg{display:block}</style>%s' "$w" "$h" "$(cat "$n.svg")" > "$html"
  chromium --headless --no-sandbox --disable-gpu --no-pdf-header-footer --print-to-pdf="$PWD/$n.pdf" "file://$html" >/dev/null 2>&1
  chromium --headless --no-sandbox --disable-gpu --hide-scrollbars --force-device-scale-factor=2 --window-size="$w,$h" --screenshot="$PWD/$n.png" "file://$html" >/dev/null 2>&1
done
