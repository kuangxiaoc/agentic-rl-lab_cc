#!/usr/bin/env bash
# Install only missing training libraries. Never reinstall/downgrade the GPU torch.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

python3 - <<'PY'
import importlib.util
import subprocess
import sys

print('preinstalled GPU torch is preserved; never run uv sync --locked on the server')
if importlib.util.find_spec('torch') is None:
    raise SystemExit('torch is missing; select the CUDA 13.0 / torch 2.13 ModelScope image')

required = {
    # transformers >= 5.17.0 adds stable qwen3 support (Qwen3 family).
    'transformers': 'transformers>=5.17.0',
    'peft': 'peft',
    'bitsandbytes': 'bitsandbytes',
    'accelerate': 'accelerate',
    'datasets': 'datasets',
    'modelscope': 'modelscope',
    'PIL': 'pillow',
}
missing = [package for module, package in required.items() if importlib.util.find_spec(module) is None]
if missing:
    print('pip install ' + ' '.join(missing), flush=True)
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', *missing])
else:
    print('all runtime imports are already installed')
PY

bash scripts/run_modelscope.sh doctor
