# ModelScope 本地 GPU 实验报告

更新时间：2026-11-10T06:25:16+00:00

| 章节 | 状态 | run_id | 说明 |
| --- | --- | --- | --- |
| 01 | completed | run-20261108-183411 | {"artifact": "/mnt/workspace/agentic-rl-lab/runs/01-loss-functions/run-20261108-183411/losses.json", "device": "cuda"} |
| 02 | completed | run-20261108-184052 | {"adapter": "/mnt/workspace/agentic-rl-lab/runs/02-grpo/run-20261108-184052/adapter", "error": "AttributeError: 'function' object has no attribute 'snapshot_download'", "error_log": "/mnt/workspace/agentic-rl-lab/runs/02-grpo/run-20261108-163919/error.txt", "exit_code": 1, "log": "/mnt/workspace/agentic-rl-lab/logs/chapter-02.log", "steps": 10, "summary": "/mnt/workspace/agentic-rl-lab/runs/02-grpo/run-20261108-184052/summary.json"} |
| 03 | completed | run-20261108-184536 | {"adapter": "/mnt/workspace/agentic-rl-lab/runs/03-vision-grpo/run-20261108-184536/adapter", "dataset": "hz2475/geoQA", "error": "AttributeError: 'Qwen3VLProcessor' object has no attribute 'eos_token_id'", "error_log": "/mnt/workspace/agentic-rl-lab/runs/03-vision-grpo/run-20261108-182408/error.txt", "exit_code": 1, "log": "/mnt/workspace/agentic-rl-lab/logs/chapter-03.log", "note": "no text-data substitute", "reason": "HF GeoQA preflight failed: LocalEntryNotFoundError: Got: ConnectError: [Errno 101] Network is unreachable\nAn error happened while trying to locate the files on the Hub, and we cannot find the appropriate snapshot folder for the specified revision on the local disk. Please check your internet connection and try again.", "steps": 10, "summary": "/mnt/workspace/agentic-rl-lab/runs/03-vision-grpo/run-20261108-184536/summary.json"} |
| 04 | completed | run-20261110-140852 | {"adapter": "/mnt/workspace/agentic-rl-lab/runs/04-pytrio-jev/run-20261110-140852/adapter", "eval_accuracy": 0.344, "steps": 40, "summary": "/mnt/workspace/agentic-rl-lab/runs/04-pytrio-jev/run-20261110-140852/summary.json"} |
| 05 | completed | run-20261110-141035 | {"adapter": "/mnt/workspace/agentic-rl-lab/runs/05-general-opd/run-20261110-141035/adapter", "steps": 15, "student_model": "Qwen/Qwen3-1.7B", "summary": "/mnt/workspace/agentic-rl-lab/runs/05-general-opd/run-20261110-141035/summary.json", "teacher_model": "Qwen/Qwen3-4B-Instruct-2507"} |
| 06 | completed | run-20261110-141706 | {"ceval_base_accuracy": 0.25, "ceval_final_accuracy": 0.225, "final_adapter": "/mnt/workspace/agentic-rl-lab/runs/06-medical-opd/run-20261110-141706/student_anchor_adapter/adapter", "medqa_base_accuracy": 0.18, "medqa_final_accuracy": 0.16, "mode": "sar", "sft_steps": 30, "teacher_adapter": "/mnt/workspace/agentic-rl-lab/runs/06-medical-opd/run-20261110-141706/medical_teacher_adapter/adapter"} |
| 07 | completed | run-20261110-142349 | {"adapter": "/mnt/workspace/agentic-rl-lab/runs/07-dapo/run-20261110-142349/adapter", "skipped_groups": 2, "steps": 2, "summary": "/mnt/workspace/agentic-rl-lab/runs/07-dapo/run-20261110-142349/summary.json"} |
| 08 | completed | run-20261110-142435 | {"adapter": "/mnt/workspace/agentic-rl-lab/runs/08-gspo/run-20261110-142435/adapter", "steps": 2, "summary": "/mnt/workspace/agentic-rl-lab/runs/08-gspo/run-20261110-142435/summary.json"} |
| 09 | not_ported | - | {} |
| 10 | not_ported | - | {} |
| 11 | not_ported | - | {} |
| 12 | not_ported | - | {} |
| 13 | not_ported | - | {} |
| 14 | not_ported | - | {} |
| 15 | not_ported | - | {} |
| 16 | not_ported | - | {} |

09-16 的 `not_ported` 是如实记录，不代表运行成功。
外部 API 预算与密钥状态见 `api_budget.json` 和 `environment.json`。
