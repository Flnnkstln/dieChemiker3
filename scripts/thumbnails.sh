#!/usr/bin/env bash
set -euo pipefail

uvx --from playwright playwright install chromium

uvx \
  --with "marimo[recommended]" \
  --with playwright \
  --with numpy \
  --with matplotlib \
  marimo export thumbnail \
  --execute \
  --overwrite \
  notebooks
