"""01 Loss Functions: local GPU tensor demonstration, in the original reading order.

No model download is needed. Run from the repository root:
    python3 experiments/loss-functions/local_gpu.py --workspace-root /mnt/workspace/agentic-rl-lab
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import local_gpu_common as common  # noqa: E402
import modelscope_lib as lib  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="本地 GPU 版 Loss Functions 张量演示")
    parser.add_argument("--workspace-root", default=None)
    parser.add_argument("--run-id", default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = lib.resolve_workspace_root(args.workspace_root)
    root.mkdir(parents=True, exist_ok=True)
    run_dir = lib.new_run_dir(root, 1, args.run_id)
    run_id = run_dir.name
    lib.initialize_state(root)
    lib.init_budget(root)
    lib.write_environment(root, REPO_ROOT, "smoke")
    lib.write_study_notes(root)
    lib.update_chapter_state(root, 1, "running", run_id)

    # Imported lazily: unit tests and --help do not require torch.
    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    old_logprob = torch.tensor([-1.0, -0.6, -0.9], device=device, dtype=torch.float32)
    current_logprob = torch.tensor(
        [-0.5, -1.1, -0.9], device=device, dtype=torch.float32, requires_grad=True
    )
    advantage = torch.tensor([2.0, 1.0, -1.0], device=device, dtype=torch.float32)
    ratio = torch.exp(current_logprob - old_logprob)
    clip_high = 1.2
    clipped_ratio = torch.clamp(ratio, 1.0 / clip_high, clip_high).detach()
    tensor_losses = {
        "importance_sampling": -(ratio * advantage),
        "ppo": -torch.minimum(ratio * advantage, clip_high * advantage),
        "cispo": -(clipped_ratio * current_logprob * advantage),
    }
    scalar_examples = [
        common.scalar_losses(-1.0, -0.5, 2.0),
        common.scalar_losses(-0.6, -1.1, 1.0),
        common.scalar_losses(-0.9, -0.9, -1.0),
    ]
    report = {
        "device": str(device),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "inputs": {
            "sampling_logprobs": old_logprob.tolist(),
            "target_logprobs": [-0.5, -1.1, -0.9],
            "advantages": advantage.tolist(),
        },
        "per_token_losses": {
            key: value.detach().cpu().tolist() for key, value in tensor_losses.items()
        },
        "scalar_examples": scalar_examples,
        "study_order": [
            "experiments/loss-functions/readme.md",
            "experiments/loss-functions/local_gpu.py",
            "experiments/grpo/readme.md",
        ],
    }
    gradients: dict[str, list[float]] = {}
    for name, values in tensor_losses.items():
        current_logprob.grad = None
        values.sum().backward(retain_graph=True)
        gradients[name] = current_logprob.grad.detach().cpu().tolist()
    report["gradients"] = gradients
    (run_dir / "losses.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    lib.append_metrics(root, 1, run_id, {"losses": report["per_token_losses"], "device": str(device)})
    lib.update_chapter_state(
        root,
        1,
        "completed",
        run_id,
        details={"artifact": str(run_dir / "losses.json"), "device": str(device)},
    )
    lib.write_report(root)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
