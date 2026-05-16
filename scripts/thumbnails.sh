#!/usr/bin/env bash
set -euo pipefail

uvx --from playwright playwright install chromium

uvx \
  --with "marimo[recommended]" \
  --with playwright \
  --with numpy \
  --with matplotlib \
  --with pandas \
  --with scipy \
  --with plotly \
  marimo export thumbnail \
  --execute \
  --overwrite \
  notebooks
