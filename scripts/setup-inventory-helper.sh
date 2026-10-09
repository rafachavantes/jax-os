#!/usr/bin/env bash
# Idempotent setup for the isolated inventory write helper.
#
# Usage: scripts/setup-inventory-helper.sh <absolute-venv-dir>
#
# Creates the venv when missing, then installs the pinned requirements into
# THAT venv only. No sudo, no global pip, no environment-file edits, no
# auto-install from HTTP handlers or startup hooks.
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 <absolute-venv-dir>" >&2
  exit 2
fi

dir="$1"
if [[ "$dir" != /* ]]; then
  echo "refusing non-absolute venv dir: $dir" >&2
  exit 2
fi

if [[ -L "$dir" ]]; then
  echo "refusing symlink venv dir: $dir" >&2
  exit 2
fi

if [[ -e "$dir" && ! -d "$dir" ]]; then
  echo "refusing: $dir exists and is not a directory" >&2
  exit 2
fi

if [[ -d "$dir" ]]; then
  if [[ "$(stat -c %u "$dir")" != "$(id -u)" ]]; then
    echo "refusing unowned venv dir: $dir" >&2
    exit 2
  fi
  if [[ ! -f "$dir/pyvenv.cfg" ]]; then
    echo "refusing: $dir exists and is not a virtualenv" >&2
    exit 2
  fi
else
  python3 -m venv "$dir"
fi

"$dir/bin/python" -m pip install --disable-pip-version-check -r "$(dirname "$0")/requirements-inventory.txt"
echo "inventory helper ready: $dir"
