#!/usr/bin/env bash
# Refresh durable reports/notes and create one downloadable tar.gz.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
export WORKSPACE_ROOT="${WORKSPACE_ROOT:-/mnt/workspace/agentic-rl-lab}"

python3 - "$WORKSPACE_ROOT" "$REPO_ROOT" <<'PY'
import sys
from pathlib import Path
sys.path.insert(0, str(Path('scripts').resolve()))
import modelscope_lib as lib

root = lib.resolve_workspace_root(sys.argv[1])
repo = Path(sys.argv[2]).resolve()
lib.initialize_state(root)
lib.init_budget(root)
lib.write_environment(root, repo, 'export')
lib.write_study_notes(root)
lib.write_report(root)
archive = lib.create_archive(root, repo)
print(f'archive={archive}')
print(f'download_dir={archive.parent}')
PY
