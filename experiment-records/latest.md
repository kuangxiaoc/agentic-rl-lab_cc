# ModelScope 本地 GPU 实验报告

更新时间：2026-11-11T15:37:15+00:00

| 章节 | 状态 | run_id | 说明 |
| --- | --- | --- | --- |
| 01 | completed | run-20261108-183411 | {"artifact": "/mnt/workspace/agentic-rl-lab/runs/01-loss-functions/run-20261108-183411/losses.json", "device": "cuda"} |
| 02 | completed | run-20261110-225538 | {"adapter": "/mnt/workspace/agentic-rl-lab/runs/02-grpo/run-20261110-225538/adapter", "attempts": 34, "skipped_groups": 30, "steps": 4, "summary": "/mnt/workspace/agentic-rl-lab/runs/02-grpo/run-20261110-225538/summary.json"} |
| 03 | completed | run-20261110-231624 | {"adapter": "/mnt/workspace/agentic-rl-lab/runs/03-vision-grpo/run-20261110-231624/adapter", "attempts": 6, "skipped_groups": 4, "steps": 2, "summary": "/mnt/workspace/agentic-rl-lab/runs/03-vision-grpo/run-20261110-231624/summary.json"} |
| 04 | completed | run-20261110-140852 | {"adapter": "/mnt/workspace/agentic-rl-lab/runs/04-pytrio-jev/run-20261110-140852/adapter", "eval_accuracy": 0.344, "steps": 40, "summary": "/mnt/workspace/agentic-rl-lab/runs/04-pytrio-jev/run-20261110-140852/summary.json"} |
| 05 | completed | run-20261110-141035 | {"adapter": "/mnt/workspace/agentic-rl-lab/runs/05-general-opd/run-20261110-141035/adapter", "steps": 15, "student_model": "Qwen/Qwen3-1.7B", "summary": "/mnt/workspace/agentic-rl-lab/runs/05-general-opd/run-20261110-141035/summary.json", "teacher_model": "Qwen/Qwen3-4B-Instruct-2507"} |
| 06 | completed | run-20261110-141706 | {"ceval_base_accuracy": 0.25, "ceval_final_accuracy": 0.225, "final_adapter": "/mnt/workspace/agentic-rl-lab/runs/06-medical-opd/run-20261110-141706/student_anchor_adapter/adapter", "medqa_base_accuracy": 0.18, "medqa_final_accuracy": 0.16, "mode": "sar", "sft_steps": 30, "teacher_adapter": "/mnt/workspace/agentic-rl-lab/runs/06-medical-opd/run-20261110-141706/medical_teacher_adapter/adapter"} |
| 07 | completed | run-20261110-233703 | {"adapter": "/mnt/workspace/agentic-rl-lab/runs/07-dapo/run-20261110-233703/adapter", "attempts": 27, "skipped_groups": 24, "steps": 3, "summary": "/mnt/workspace/agentic-rl-lab/runs/07-dapo/run-20261110-233703/summary.json"} |
| 08 | completed | run-20261111-214640 | {"adapter": "/mnt/workspace/agentic-rl-lab/runs/08-gspo/run-20261111-214640/adapter", "attempts": 23, "skipped_groups": 20, "steps": 3, "summary": "/mnt/workspace/agentic-rl-lab/runs/08-gspo/run-20261111-214640/summary.json"} |
| 09 | completed | run-20261110-150419 | {"adapter": "/mnt/workspace/agentic-rl-lab/runs/09-opsd/run-20261110-150419/adapter", "steps": 4, "summary": "/mnt/workspace/agentic-rl-lab/runs/09-opsd/run-20261110-150419/summary.json"} |
| 10 | failed | run-20261111-215724 | {"deferred": true, "interrupted": "/mnt/workspace/agentic-rl-lab/runs/10-agentopsd/run-20261111-215724/interrupted.txt", "log": "/mnt/workspace/agentic-rl-lab/logs/chapter-10.log", "optimizer_step_observed": false, "reason": "stopped after 63 minutes without a recorded optimizer update", "run_id": "run-20261111-215724"} |
| 11 | completed | search-offline-2step-20261111 | {"adapter": "/mnt/workspace/agentic-rl-lab/runs/11-search-r1/search-offline-2step-20261111/adapter", "effective_updates": 2, "mechanism_only": true, "search_backend": "offline-evidence-oracle", "steps": 2, "summary": "/mnt/workspace/agentic-rl-lab/runs/11-search-r1/search-offline-2step-20261111/summary.json", "updated_trajectories": 8} |
| 12 | completed | retool-2step-20261111 | {"adapter": "/mnt/workspace/agentic-rl-lab/runs/12-retool/retool-2step-20261111/adapter", "effective_updates": 2, "mechanism_only": true, "steps": 2, "summary": "/mnt/workspace/agentic-rl-lab/runs/12-retool/retool-2step-20261111/summary.json", "updated_trajectories": 8} |
| 13 | completed | run-alfworld-grpo-smoke-20261110 | {"adapter": "/mnt/workspace/agentic-rl-lab/runs/13-alfworld/run-alfworld-grpo-smoke-20261110/adapter", "mechanism_only": true, "reward": "task success - 0.1 * invalid_action_count", "steps": 1, "summary": "/mnt/workspace/agentic-rl-lab/runs/13-alfworld/run-alfworld-grpo-smoke-20261110/summary.json"} |
| 14 | completed | run-tempo-smoke-20261110 | {"actor_critic": "same adapter, different prompts", "adapter": "/mnt/workspace/agentic-rl-lab/runs/14-tempo/run-tempo-smoke-20261110/adapter", "mechanism_only": true, "summary": "/mnt/workspace/agentic-rl-lab/runs/14-tempo/run-tempo-smoke-20261110/summary.json", "updates": 2} |
| 15 | completed | run-spec-o3-both-sft20-r5 | {"mechanism_only": true, "rl_steps": 2, "sft_steps": 20, "summary": "/mnt/workspace/agentic-rl-lab/runs/15-spec-o3/run-spec-o3-both-sft20-r5/summary.json"} |
| 16 | completed | run-tournament-smoke-r6 | {"mechanism_only": true, "steps": 1, "summary": "/mnt/workspace/agentic-rl-lab/runs/16-tournament-grpo/run-tournament-smoke-r6/summary.json", "tavily_calls": 2} |

每章状态来自实际运行；没有真实运行证据时保持 `not_ported` 或 `data_unavailable`。
`completed` 只表示该 run 正常结束，不表示已经达到论文规模；10–16 当前是 smoke / 机制运行。
第 10–12 章必须检查 `effective_updates` / `updated_trajectories`；为零时不能视为策略更新成功。
外部 API 预算与密钥状态见 `api_budget.json` 和 `environment.json`。
