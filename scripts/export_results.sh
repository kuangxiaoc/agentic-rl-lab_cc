#!/usr/bin/env bash
# Refresh durable reports/notes and create one downloadable tar.gz of results only.
# Sources live in Git; pass --with-code for a self-contained snapshot.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
export WORKSPACE_ROOT="${WORKSPACE_ROOT:-/mnt/workspace/agentic-rl-lab}"

INCLUDE_CODE=0
for arg in "$@"; do
    case "$arg" in
        --with-code) INCLUDE_CODE=1 ;;
        *) echo "unknown argument: $arg (only --with-code is supported)" >&2; exit 2 ;;
    esac
done

python3 - "$WORKSPACE_ROOT" "$REPO_ROOT" "$INCLUDE_CODE" <<'PY'
import sys
from pathlib import Path
sys.path.insert(0, str(Path('scripts').resolve()))
import modelscope_lib as lib

root = lib.resolve_workspace_root(sys.argv[1])
repo = Path(sys.argv[2]).resolve()
include_code = sys.argv[3] == '1'
lib.initialize_state(root)
lib.init_budget(root)
lib.write_environment(root, repo, 'export')
lib.write_study_notes(root)
lib.write_report(root)
copied = lib.sync_results_to_repo(root, repo)
for chapter, destination in copied.items():
    print(f'synced_chapter_{chapter:02d}={destination.relative_to(repo)}')
archive = lib.create_archive(root, repo, include_code=include_code)
print(f'archive={archive}')
print(f'download_dir={archive.parent}')
print(f'contents={"results+code" if include_code else "results-only (code is in Git)"}')
PY
