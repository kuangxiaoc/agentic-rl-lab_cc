"""Sequential chapter runner for the persistent ModelScope GPU workspace."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

import modelscope_lib as lib  # noqa: E402


CHAPTER_SCRIPTS = {
    1: REPO_ROOT / "experiments" / "loss-functions" / "local_gpu.py",
    2: REPO_ROOT / "experiments" / "grpo" / "local_gpu.py",
    3: REPO_ROOT / "experiments" / "vision-grpo" / "local_gpu.py",
    4: REPO_ROOT / "experiments" / "dapo" / "local_gpu.py",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="按阅读顺序运行 ModelScope 本地 GPU 实验")
    parser.add_argument("command", nargs="?", default="run", choices=("run", "chapter", "doctor"))
    parser.add_argument("chapters", nargs="?", default=None, help="例如 1-3 / 02 / all")
    parser.add_argument("--workspace-root", default=None)
    parser.add_argument("--profile", default=os.environ.get("RUN_PROFILE", "smoke"), choices=("smoke", "full"))
    parser.add_argument("--force", action="store_true", help="重跑已完成的章节")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--skip-model-downloads", action="store_true", help="diagnostic only: fail 02/03 before downloads")
    parser.add_argument(
        "--chapter-arg",
        action="append",
        default=[],
        metavar="ARG",
        help="extra flag forwarded to every chapter script, for example "
        "--chapter-arg=--dynamic-sampling",
    )
    return parser.parse_args()


def doctor(root: Path, profile: str) -> int:
    """Check persistent paths and the preinstalled GPU stack without downloading a model."""
    lib.initialize_state(root)
    lib.init_budget(root)
    lib.write_environment(root, REPO_ROOT, profile)
    lib.write_study_notes(root)
    lib.write_report(root)
    print(f"workspace={root}")
    print(f"profile={profile}")
    print(f"python={sys.version.split()[0]}")
    try:
        import torch

        print(f"torch={torch.__version__} cuda_available={torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"gpu={torch.cuda.get_device_name(0)}")
            props = torch.cuda.get_device_properties(0)
            print(
                f"vram={props.total_memory / 1024**3:.1f}GiB "
                f"capability={torch.cuda.get_device_capability(0)}"
            )
    except ImportError as exc:
        print(f"torch=missing ({exc})")
    for name in ("transformers", "peft", "bitsandbytes", "modelscope", "datasets", "accelerate"):
        try:
            module = __import__(name)
            print(f"{name}={getattr(module, '__version__', 'installed')}")
        except ImportError as exc:
            print(f"{name}=missing ({exc})")
    # 4-bit loading is the step that can fail with an intact-looking install, so
    # report the compiled extension state before any chapter spends time on a download.
    try:
        from bitsandbytes import cextension

        setup = getattr(cextension, "setup", None)
        print(f"bitsandbytes_cuda={getattr(setup, 'available', 'unknown')}")
    except Exception as exc:  # noqa: BLE001 - diagnostics must never fail the doctor
        print(f"bitsandbytes_cuda=error ({type(exc).__name__}: {exc})")
    print(f"api_key_present DASHSCOPE_API_KEY={bool(os.environ.get('DASHSCOPE_API_KEY'))}")
    print("external_api_calls=disabled (budget reservation only)")
    return 0


def log_tail(log_path: Path, lines: int = 25) -> list[str]:
    """Read the end of a chapter log; the failure cause lives here, not on stdout."""
    try:
        return log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-lines:]
    except OSError as exc:
        return [f"could not read {log_path}: {type(exc).__name__}: {exc}"]


def run(
    root: Path,
    chapters: list[int],
    profile: str,
    force: bool,
    continue_on_error: bool,
    skip_model_downloads: bool = False,
    chapter_args: list[str] | None = None,
) -> int:
    lib.initialize_state(root)
    lib.init_budget(root)
    lib.write_environment(root, REPO_ROOT, profile)
    lib.write_study_notes(root)
    failures: list[int] = []
    for chapter in chapters:
        state = lib.load_state(root)
        if lib.should_skip_chapter(state, chapter, force):
            print(f"[skip] chapter {chapter:02d}: {state['chapters'][str(chapter)]['status']}")
            continue
        script = CHAPTER_SCRIPTS.get(chapter)
        if script is None:
            previous = state["chapters"][str(chapter)]
            status = previous["status"] if previous["status"] == "not_ported" else "skipped"
            lib.update_chapter_state(
                root,
                chapter,
                status,
                details={"reason": "not_ported to local GPU", "fake_success": False},
            )
            print(f"[skip] chapter {chapter:02d}: {status}, no fake success")
            continue
        lib.update_chapter_state(root, chapter, "running")
        log_path = root / "logs" / f"chapter-{chapter:02d}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        env = os.environ.copy()
        env.update(
            {
                "WORKSPACE_ROOT": str(root),
                "RUN_PROFILE": profile,
                "RUN_CHAPTERS": str(chapter),
                "PYTHONUNBUFFERED": "1",
            }
        )
        if skip_model_downloads and chapter > 1:
            env["LOCAL_GPU_SKIP_DOWNLOADS"] = "1"
        print(f"[run ] chapter {chapter:02d}: {script.relative_to(REPO_ROOT)} -> {log_path}")
        with log_path.open("a", encoding="utf-8") as log:
            result = subprocess.run(
                [
                    sys.executable,
                    str(script),
                    "--workspace-root",
                    str(root),
                    *(chapter_args or []),
                ],
                cwd=REPO_ROOT,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=False,
            )
        state = lib.load_state(root)
        status = state["chapters"][str(chapter)]["status"]
        if result.returncode != 0 and status not in {"completed", "data_unavailable"}:
            lib.update_chapter_state(
                root, chapter, "failed", details={"exit_code": result.returncode, "log": str(log_path)}
            )
            failures.append(chapter)
            print(f"[fail] chapter {chapter:02d}: exit={result.returncode}, log={log_path}")
            for line in log_tail(log_path):
                print(f"    {line[-200:]}")
            if not continue_on_error:
                break
        else:
            print(f"[done] chapter {chapter:02d}: {status}")
    lib.write_report(root)
    copied = lib.sync_results_to_repo(root, REPO_ROOT)
    for chapter, destination in copied.items():
        print(f"[sync] chapter {chapter:02d}: {destination.relative_to(REPO_ROOT)}")
    if failures:
        print(f"failed chapters: {failures}")
        print("fix hint: read the traceback above, or the synced copy under the chapter results dir")
        return 1
    return 0


def main() -> int:
    args = parse_args()
    root = lib.resolve_workspace_root(args.workspace_root)
    root.mkdir(parents=True, exist_ok=True)
    spec = args.chapters or os.environ.get("RUN_CHAPTERS", "1-3")
    try:
        chapters = lib.parse_chapter_spec(spec)
    except ValueError as exc:
        print(f"Invalid chapter spec {spec!r}: {exc}", file=sys.stderr)
        return 2
    if args.command == "doctor":
        return doctor(root, args.profile)
    if args.command == "chapter" and len(chapters) != 1:
        print("The `chapter` command accepts exactly one chapter (for example: chapter 02)", file=sys.stderr)
        return 2
    return run(
        root,
        chapters,
        args.profile,
        args.force,
        args.continue_on_error,
        args.skip_model_downloads,
        args.chapter_arg,
    )


if __name__ == "__main__":
    raise SystemExit(main())
