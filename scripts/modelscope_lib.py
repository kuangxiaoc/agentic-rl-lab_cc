"""Persistence, budget, archive, and report helpers for the ModelScope GPU fork.

This module only imports the standard library. Chapter scripts import training
dependencies lazily so local syntax/unit checks never need a GPU stack.
"""

from __future__ import annotations

import json
import io
import os
import re
import shutil
import tarfile
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any


DEFAULT_WORKSPACE_ROOT = Path("/mnt/workspace/agentic-rl-lab")
CHAPTERS = range(1, 17)
CHAPTER_NAMES = {
    1: "01-loss-functions",
    2: "02-grpo",
    3: "03-vision-grpo",
    4: "04-pytrio-jev",
    5: "05-general-opd",
    6: "06-medical-opd",
    7: "07-dapo",
    8: "08-gspo",
    9: "09-opsd",
    10: "10-agentopsd",
    11: "11-search-r1",
    12: "12-retool",
    13: "13-alfworld",
    14: "14-tempo",
    15: "15-spec-o3",
    16: "16-tournament-grpo",
}
CHAPTER_RESULT_DIRS = {
    1: "experiments/loss-functions/results",
    2: "experiments/grpo/results",
    3: "experiments/vision-grpo/results",
    4: "experiments/pytrio-jev/results",
    5: "experiments/general-opd/results",
    6: "experiments/medical-opd/results",
    7: "experiments/dapo/results",
    8: "experiments/gspo/results",
    9: "experiments/opsd/results",
    10: "experiments/agentopsd/results",
    11: "experiments/search-r1/results",
    12: "experiments/retool/results",
    13: "experiments/alfworld/results",
    14: "experiments/tempo/results",
    15: "experiments/spec-o3/results",
    16: "experiments/tournament-grpo/results",
}
VALID_STATUSES = {
    "not_ported",
    "pending",
    "running",
    "completed",
    "failed",
    "data_unavailable",
    "skipped",
}


class ExternalBudgetError(RuntimeError):
    """Raised before a paid external API call would exceed a hard limit."""


def parse_chapter_spec(spec: str | int | None) -> list[int]:
    """Parse `1-3`, `4-6,8`, `02` or `all` into the canonical reading order."""
    text = "all" if spec is None or str(spec).strip() == "" else str(spec).strip().lower()
    if text in {"all", "1-16"}:
        return list(CHAPTERS)
    chapters: set[int] = set()
    for part in text.split(","):
        part = part.strip()
        if not part:
            raise ValueError(f"Empty chapter in spec: {spec!r}")
        if "-" in part:
            first, last = part.split("-", 1)
            start, end = int(first), int(last)
            if start > end:
                raise ValueError(f"Descending chapter range: {part}")
            chapters.update(range(start, end + 1))
        else:
            chapters.add(int(part))
    if not chapters or min(chapters) < 1 or max(chapters) > 16:
        raise ValueError(f"Chapter spec must be within 1-16: {spec!r}")
    return sorted(chapters)


def resolve_workspace_root(value: str | os.PathLike[str] | None = None) -> Path:
    """Resolve the one persistent root; `/` is rejected to avoid archive accidents."""
    raw = os.environ.get("WORKSPACE_ROOT") if value is None else os.fspath(value)
    root = Path(raw or DEFAULT_WORKSPACE_ROOT).expanduser().resolve()
    if root == Path(root.anchor):
        raise ValueError(f"Refusing unsafe workspace root: {root}")
    return root


def ensure_within_root(root: Path, candidate: str | os.PathLike[str]) -> Path:
    """Resolve an output path and reject symlink/path traversal outside root."""
    root = Path(root).resolve()
    target = Path(candidate).resolve()
    if target != root and root not in target.parents:
        raise ValueError(f"Path escapes workspace root: {target}")
    return target


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    )
    try:
        with handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(handle.name, path)
    finally:
        if os.path.exists(handle.name):
            os.unlink(handle.name)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def initialize_state(root: Path) -> dict[str, Any]:
    """Create state.json once and return its current contents."""
    path = Path(root) / "state.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    state = {
        "schema_version": 1,
        "created_at": _now(),
        "updated_at": _now(),
        "chapters": {
            str(index): {
                "status": "not_ported",
                "run_id": None,
                "details": {},
                "updated_at": _now(),
            }
            for index in CHAPTERS
        },
    }
    _atomic_json(path, state)
    return state


def load_state(root: Path) -> dict[str, Any]:
    return initialize_state(root)


def update_chapter_state(
    root: Path,
    chapter: int,
    status: str,
    run_id: str | None = None,
    details: dict[str, Any] | None = None,
    *,
    replace_details: bool = False,
) -> dict[str, Any]:
    """Atomically record one chapter's status and merge its details."""
    if status not in VALID_STATUSES:
        raise ValueError(f"Invalid chapter status: {status}")
    if chapter not in CHAPTERS:
        raise ValueError(f"Invalid chapter: {chapter}")
    state = initialize_state(root)
    entry = state["chapters"][str(chapter)]
    merged_details = {} if replace_details else dict(entry.get("details", {}))
    merged_details.update(details or {})
    entry.update(
        {
            "status": status,
            "run_id": run_id,
            "updated_at": _now(),
            "details": merged_details,
        }
    )
    state["updated_at"] = _now()
    _atomic_json(Path(root) / "state.json", state)
    return state


def should_skip_chapter(state: dict[str, Any], chapter: int, force: bool = False) -> bool:
    if force:
        return False
    status = state["chapters"][str(chapter)]["status"]
    return status in {"completed", "data_unavailable"}


def init_budget(root: Path) -> dict[str, Any]:
    """Initialize the hard external-API limits used by future online chapters."""
    path = Path(root) / "api_budget.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    payload = {
        "schema_version": 1,
        "created_at": _now(),
        "updated_at": _now(),
        "max_external_cost_cny": float(os.environ.get("MAX_EXTERNAL_COST_CNY", "50")),
        "allow_paid_overage": os.environ.get("ALLOW_PAID_OVERAGE", "0") == "1",
        "estimated_cost_cny": 0.0,
        "free_token_limit": int(os.environ.get("FREE_TOKEN_LIMIT", "1000000")),
        "providers": {},
        "models": {},
        "events": [],
    }
    _atomic_json(path, payload)
    return payload


def record_external_usage(
    root: Path,
    provider: str,
    model: str,
    input_tokens: int,
    output_tokens: int,
    estimated_cost_cny: float = 0.0,
    allow_unlisted_model: bool = False,
) -> dict[str, Any]:
    """Reject a call before it can exceed free tokens or the CNY hard cap."""
    budget = init_budget(root)
    if not allow_unlisted_model and model not in {"qwen3.8-flash", "deepseek-v4.1-flash"}:
        raise ExternalBudgetError(f"Model is not on the default low-cost allowlist: {model}")
    tokens = int(input_tokens) + int(output_tokens)
    model_entry = budget["models"].get(
        model, {"tokens": 0, "input_tokens": 0, "output_tokens": 0}
    )
    if model_entry["tokens"] + tokens > budget["free_token_limit"] and not budget["allow_paid_overage"]:
        raise ExternalBudgetError(f"Free token quota would be exceeded for {model}")
    if budget["estimated_cost_cny"] + estimated_cost_cny > budget["max_external_cost_cny"]:
        raise ExternalBudgetError("External API cost cap would be exceeded")
    model_entry["tokens"] += tokens
    model_entry["input_tokens"] += int(input_tokens)
    model_entry["output_tokens"] += int(output_tokens)
    budget["models"][model] = model_entry
    provider_entry = budget["providers"].get(provider, {"tokens": 0, "calls": 0})
    provider_entry["tokens"] += tokens
    provider_entry["calls"] += 1
    budget["providers"][provider] = provider_entry
    budget["estimated_cost_cny"] += float(estimated_cost_cny)
    budget["events"].append(
        {
            "at": _now(),
            "provider": provider,
            "model": model,
            "input_tokens": int(input_tokens),
            "output_tokens": int(output_tokens),
            "estimated_cost_cny": float(estimated_cost_cny),
        }
    )
    budget["updated_at"] = _now()
    _atomic_json(Path(root) / "api_budget.json", budget)
    return budget


_SECRET_PATTERNS = [
    re.compile(r"(Bearer\s+)\S+", re.IGNORECASE),
    re.compile(r"((?:API[_-]?KEY|TOKEN|SECRET|PASSWORD)\s*[=:]\s*)\S+", re.IGNORECASE),
    re.compile(r"\{\{[^}]+\}\}"),
]


def redact_text(text: str) -> str:
    """Mask obvious credentials and template placeholders in logs/reports."""
    for pattern in _SECRET_PATTERNS:
        text = pattern.sub(lambda match: match.group(1) + "[REDACTED]" if match.lastindex else "[REDACTED]", text)
    return text


def _archive_filter(info: tarfile.TarInfo) -> tarfile.TarInfo | None:
    name = PurePosixPath(info.name)
    if any(part in {".env", ".git", "__pycache__"} or part.endswith(".pyc") for part in name.parts):
        return None
    if any(part.startswith(".env.") for part in name.parts):
        return None
    if len(name.parts) > 1 and name.parts[0] == "cache":
        return None
    if info.isfile() and info.size > 100 * 1024 * 1024:
        return None
    info.uid = info.gid = 0
    info.uname = info.gname = ""
    return info


def _sync_filter(info: tarfile.TarInfo) -> tarfile.TarInfo | None:
    """Mirror workspace records into git while excluding secrets and bulky weights."""
    if _archive_filter(info) is None:
        return None
    name = PurePosixPath(info.name)
    if name.suffix in {".safetensors", ".bin", ".pt", ".pth", ".ckpt", ".parquet", ".tar", ".gz"}:
        return None
    if info.isfile() and info.size > 2 * 1024 * 1024:
        return None
    return info


def _copy_synced_file(source: Path, target: Path) -> None:
    """Copy one small artifact, redacting obvious secrets in text records."""
    target.parent.mkdir(parents=True, exist_ok=True)
    if source.suffix in {".json", ".jsonl", ".log", ".md", ".txt"}:
        target.write_text(
            redact_text(source.read_text(encoding="utf-8", errors="replace")),
            encoding="utf-8",
        )
    else:
        shutil.copy2(source, target)


def create_archive(root: Path, repo: Path, include_code: bool = False) -> Path:
    """Package persistent results for local download.

    Code is not included by default: the same sources are already in Git, and a
    second copy inside the archive drifts from the committed version. Pass
    ``include_code=True`` only when you want a self-contained snapshot.
    """
    root = Path(root).resolve()
    repo = Path(repo).resolve()
    ensure_within_root(root, root)
    exports = root / "exports"
    exports.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    archive = exports / f"agentic-rl-lab-{stamp}.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        tar.add(root / "state.json", arcname="results/state.json", filter=_archive_filter)
        for name in ("runs", "logs", "reports", "metrics", "api_budget.json", "environment.json", "study_notes.md"):
            source = root / name
            if source.exists():
                tar.add(source, arcname=f"results/{name}", filter=_archive_filter)
        if include_code:
            tar.add(repo, arcname="code", filter=_archive_filter)
        readme = (
            "# 实验归档\n\n"
            f"生成时间：{_now()}\n\n"
            "- `results/`：状态、指标、日志、检查点、报告和学习笔记\n"
            "- 源码不在此归档内；同一份代码在 Git 仓库中，`git pull` 即可获得\n"
            "- 模型缓存、`.env`、`.git` 与超过 100 MiB 的文件不会进入归档\n"
        )
        content = readme.encode("utf-8")
        info = tarfile.TarInfo("README.md")
        info.size = len(content)
        info.mtime = int(time.time())
        tar.addfile(info, io.BytesIO(content))
    return archive


def sync_results_to_repo(root: Path, repo: Path) -> dict[int, Path]:
    """Copy chapter records/logs/metrics under each chapter's git-visible results dir.

    Persistent heavy artifacts stay in `/mnt/workspace`; this copy is intentionally
    limited to readable JSON/Markdown/text files so `git commit && git push` is safe.
    """
    root = Path(root).resolve()
    repo = Path(repo).resolve()
    copied: dict[int, Path] = {}
    for chapter, relative in CHAPTER_RESULT_DIRS.items():
        destination = ensure_within_root(repo, repo / relative)
        destination.mkdir(parents=True, exist_ok=True)
        copied[chapter] = destination
        for source_name in (
            f"runs/{CHAPTER_NAMES[chapter]}",
            f"logs/chapter-{chapter:02d}.log",
            f"metrics/chapter-{chapter:02d}.jsonl",
        ):
            source = root / source_name
            if not source.exists():
                continue
            if source.is_file():
                _copy_synced_file(source, destination / source.name)
                continue
            for item in sorted(source.rglob("*")):
                if item.is_dir():
                    continue
                info = tarfile.TarInfo(str(item.relative_to(source)))
                info.size = item.stat().st_size
                if _sync_filter(info) is None:
                    continue
                target = destination / CHAPTER_NAMES[chapter] / item.relative_to(source)
                _copy_synced_file(item, target)
        readme = (
            f"# Chapter {chapter:02d} 本地 GPU 实验记录\n\n"
            f"本目录由 `/mnt/workspace/agentic-rl-lab` 自动同步，适合 `git commit && git push` "
            f"后拉回本地阅读。模型大权重、adapter `.safetensors`、数据集 parquet、缓存和 `.env` "
            f"不会复制到 Git；完整大产物仍在持久化目录，或使用 `bash scripts/export_results.sh` 下载归档。\n"
        )
        (destination / "README.md").write_text(readme, encoding="utf-8")
    shared = repo / "experiment-records"
    shared.mkdir(parents=True, exist_ok=True)
    for source_name in ("state.json", "api_budget.json", "environment.json", "study_notes.md"):
        source = root / source_name
        if source.exists():
            _copy_synced_file(source, shared / source_name)
    report_source = root / "reports" / "latest.md"
    if report_source.exists():
        _copy_synced_file(report_source, shared / "latest.md")
    (shared / "README.md").write_text(
        "# 全局实验记录\n\n"
        "本目录由 ModelScope GPU runner 自动同步，包含持久状态、预算、环境快照、"
        "学习笔记与最新报告。真实密钥和模型权重不会写入。\n",
        encoding="utf-8",
    )
    return copied


def write_study_notes(root: Path) -> Path:
    """Write a Chinese reading map connecting the original and local GPU code."""
    path = Path(root) / "study_notes.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        """# 本地 GPU 学习笔记

原版借助 PyTRIO 把采样、forward/backward 和优化器更新放在云端；本 fork 保留原版章节，
在每个原章节目录新增 `local_gpu.py`，把同一套 reward / advantage / loss 放到 ModelScope
单卡 GPU 上亲手实现。按 `experiments/README.md` 的阅读顺序看：

1. `experiments/loss-functions/readme.md` → `local_gpu.py`：用张量复现 IS / PPO / CISPO。
2. `experiments/grpo/readme.md` → `03-grpo.py` → `local_gpu.py`：对比 PyTRIO Datum 与
   本地 `model.generate`、旧策略 logprob、组内 advantage、PEFT 梯度更新。
3. `experiments/vision-grpo/readme.md` → `train.py` → `local_gpu.py`：观察图片如何进入
   processor、旧版 chunk 对齐如何变成本地 attention mask 与 pixel values。
4. `experiments/pytrio-jev/readme.md` → `train.py` → `local_gpu.py`：对比 PyTRIO 的
   marker-only masked SFT 与本地 QLoRA 单次 forward 分类。
5. `experiments/general-opd/readme.md` → `01-demo-sync.py` → `local_gpu.py`：对比
   `compute_logprobs` 与本地 teacher forward，亲手算 reverse KL。
6. `experiments/medical-opd/readme.md` → `03/04/05-*.py` → `local_gpu.py`：把
   Medical SFT、Medical OPD、Base-anchor OPD 和可选 IDT 串成一条本地训练链。
7. `experiments/dapo/readme.md` → `train.py` → `local_gpu.py`：对照 Dynamic Sampling
   和 token-level loss 在本地如何落到 `model.generate` 与 `backward()`。
8. `experiments/gspo/readme.md` → `train.py` → `local_gpu.py`：对照 sequence-level
   importance ratio 与本地整条 completion 的 log-ratio。
9. `experiments/opsd/readme.md` → `01-opsd-sync.py` → `local_gpu.py`：Student 只
   看题，固定 Teacher 看题和参考解；两者对同一条 completion 的 logprob 差就是
   sampled-token reverse KL。
10. `experiments/agentopsd/readme.md` → `rollout.py` → `teacher.py` →
    `local_gpu.py`：真实 ALFWorld 多轮 rollout、同快照 Self-Teacher 与
    turn-level credit。
11. `experiments/search-r1/readme.md` → `rollout.py` → `local_gpu.py`：多轮搜索
    state machine、tool observation mask 和组内 advantage；服务器无外网时使用
    明确标注的离线证据后端，不能把它当成在线搜索指标。
12. `experiments/retool/readme.md` → `protocol.py` → `sandbox.py` → `rollout.py`
    → `local_gpu.py`：代码交织轨迹、真实 Python 沙箱、observation mask 和本地
    PPO-style 更新。
13. `experiments/alfworld/readme.md` → `rollout.py` → `advantages.py` →
    `local_gpu.py`：真实 TextWorld 组内 GRPO。
14. `experiments/tempo/readme.md` → `states.py` → `critic.py` → `tempo.py` →
    `local_gpu.py`：macro-step actor/critic 与生成式 critic。
15. `experiments/spec-o3/readme.md` → `protocol.py` → `tools.py` → `local_gpu.py`：
    图文 SFT 与多轮光谱工具 GRPO。
16. `experiments/tournament-grpo/readme.md` → `reward.py` → `judge.py` →
    `local_gpu.py`：Tavily 搜索、淘汰赛奖励和本地裁判。

每次运行的配置、stdout/stderr、metrics、LoRA adapter、`state.json` 和环境快照都在本目录。
服务器释放前执行 `bash scripts/export_results.sh`，下载 `exports/*.tar.gz` 即可离线学习。
""",
        encoding="utf-8",
    )
    return path


def write_report(root: Path, state: dict[str, Any] | None = None) -> Path:
    """Write a compact status report from the durable state file."""
    state = state or load_state(root)
    lines = [
        "# ModelScope 本地 GPU 实验报告",
        "",
        f"更新时间：{state.get('updated_at', _now())}",
        "",
        "| 章节 | 状态 | run_id | 说明 |",
        "| --- | --- | --- | --- |",
    ]
    for index in CHAPTERS:
        entry = state["chapters"][str(index)]
        detail = json.dumps(entry.get("details", {}), ensure_ascii=False, sort_keys=True)
        lines.append(
            f"| {index:02d} | {entry['status']} | {entry.get('run_id') or '-'} | {detail} |"
        )
    lines += [
        "",
        "每章状态来自实际运行；没有真实运行证据时保持 `not_ported` 或 `data_unavailable`。",
        "`completed` 只表示该 run 正常结束，不表示已经达到论文规模；10–16 当前是 smoke / 机制运行。",
        "第 10 章现有两轮都因 dynamic sampling 丢弃了全部 group，"
        "`updated_trajectories=0`，不能视为策略更新成功。",
        "外部 API 预算与密钥状态见 `api_budget.json` 和 `environment.json`。",
        "",
    ]
    reports = Path(root) / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    path = reports / "latest.md"
    path.write_text(redact_text("\n".join(lines)), encoding="utf-8")
    return path


def write_environment(root: Path, repo: Path, profile: str) -> Path:
    """Record environment facts without copying any secret values."""
    import platform
    import sys

    payload = {
        "recorded_at": _now(),
        "python": sys.version,
        "platform": platform.platform(),
        "workspace_root": str(Path(root).resolve()),
        "repo": str(Path(repo).resolve()),
        "run_profile": profile,
        "api_key_present": {
            "DASHSCOPE_API_KEY": bool(os.environ.get("DASHSCOPE_API_KEY")),
            "TAVILY_API_KEY": bool(os.environ.get("TAVILY_API_KEY")),
            "SERPAPI_API_KEY": bool(os.environ.get("SERPAPI_API_KEY")),
        },
        "api_key_values": "not_recorded",
        "limits": {
            "MAX_EXTERNAL_COST_CNY": os.environ.get("MAX_EXTERNAL_COST_CNY", "50"),
            "ALLOW_PAID_OVERAGE": os.environ.get("ALLOW_PAID_OVERAGE", "0"),
            "FREE_TOKEN_LIMIT": os.environ.get("FREE_TOKEN_LIMIT", "1000000"),
        },
    }
    path = Path(root) / "environment.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        try:
            previous = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            previous = None
        if isinstance(previous, dict):
            previous_without_time = {key: value for key, value in previous.items() if key != "recorded_at"}
            payload_without_time = {key: value for key, value in payload.items() if key != "recorded_at"}
            if previous_without_time == payload_without_time:
                return path
    _atomic_json(path, payload)
    return path


def new_run_dir(root: Path, chapter: int, run_id: str | None = None) -> Path:
    """Create and return one chapter run directory under the persistent root."""
    run_id = run_id or f"run-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    name = CHAPTER_NAMES.get(chapter, f"{chapter:02d}-chapter")
    path = ensure_within_root(root, Path(root) / "runs" / name / run_id)
    path.mkdir(parents=True, exist_ok=True)
    return path


def append_metrics(root: Path, chapter: int, run_id: str, payload: dict[str, Any]) -> Path:
    """Append one redacted JSON-lines metric row for a run."""
    path = ensure_within_root(root, Path(root) / "metrics" / f"chapter-{chapter:02d}.jsonl")
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {"at": _now(), "chapter": chapter, "run_id": run_id, **payload}
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    return path


def copy_repo_for_archive(repo: Path, destination: Path) -> None:  # pragma: no cover - helper for future export
    shutil.copytree(repo, destination, ignore=shutil.ignore_patterns(".git", "__pycache__", ".env"))
