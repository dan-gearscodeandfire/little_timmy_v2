#!/usr/bin/env bash
# Manual launcher. systemd unit can land later.
set -euo pipefail
cd "$(dirname "$0")/.."
exec ./.venv/bin/python -m booth_display.server
