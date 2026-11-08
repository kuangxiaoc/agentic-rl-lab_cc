# 第 02 章：本地 GPU 上的 GRPO / GSM8K

这一章把第 01 章的三个变量接到真实模型上：用 Qwen3-4B 对 GSM8K 数学题采样一组回答，按 `\boxed{}` 答案给 0/1 reward，用组内平均分计算 advantage，再用 4-bit QLoRA 在本地完成采样、loss 和参数更新。整条链路不再经过 PyTRIO 远端训练服务。

本章脚本：[`local_gpu.py`](./local_gpu.py)
统一启动说明：[`MODELSCOPE_LOCAL_GPU.md`](../../MODELSCOPE_LOCAL_GPU.md)
原版 PyTRIO 参考：[`03-grpo.py`](./03-grpo.py)
建议阅读顺序：`01 Loss Functions → 02 GRPO → 03 Vision GRPO`

## 0. 先跑通最小闭环

在仓库根目录执行：

```bash
bash scripts/run_modelscope.sh chapter 02
```

默认 `RUN_PROFILE=smoke`：`steps=2, batch_size=1, group_size=2, max_new_tokens=128`。首次运行会从魔搭下载：

- 模型：`Qwen/Qwen3-4B-Instruct-2507`
- 数据：`AI-ModelScope/gsm8k` 的 `main/train-00000-of-00001.parquet`

下载、缓存、日志、样本、adapter 都放在 `/mnt/workspace/agentic-rl-lab/`；runner 结束时会把轻量记录同步到 Git 范围：

```text
/mnt/workspace/agentic-rl-lab/
├── runs/02-grpo/<run_id>/
│   ├── config.json
│   ├── samples.jsonl
│   ├── history.json
│   ├── summary.json
│   └── adapter/                # LoRA 权重，留在持久目录
├── logs/chapter-02.log
└── metrics/chapter-02.jsonl

experiments/grpo/results/
├── 02-grpo/<run_id>/*.json     # 自动同步，适合 commit/push
├── chapter-02.log
└── chapter-02.jsonl
```

在服务器执行：

```bash
git add experiments/grpo/results experiment-records
git commit -m "record: local GPU GRPO smoke"
git push
```

本地 `git pull` 后就能看到配置、题目、回答、reward、advantage、loss 曲线和状态报告。`.safetensors`、模型缓存、数据 parquet、`.env` 不会进入 Git；需要完整 adapter 时用 `bash scripts/export_results.sh` 下载归档。

切换 loss 或正式配置：

```bash
# 直接调用章节脚本时可切换 loss
python3 experiments/grpo/local_gpu.py --loss-fn ppo
python3 experiments/grpo/local_gpu.py --loss-fn cispo

# runner 使用 full profile
RUN_PROFILE=full bash scripts/run_modelscope.sh chapter 02
```

## 1. GRPO 解决什么问题

PPO 通常需要 value model 估计 baseline；对 LLM 来说，这个 value model 很大，而且数学题经常只有最终答案可以打分，逐 token 估值并不自然。GRPO 的做法是：

1. 对同一道题采样 `G` 个回答。
2. 给每个回答打一个可验证 reward。
3. 用组内平均 reward 当 baseline。
4. 比平均分好的回答 advantage 为正，差的为负。
5. 用这个 advantage 更新 policy，不训练 value model。

公式是：

$$
A_i = r_i - \mathrm{mean}(r_1, r_2, \ldots, r_G)
$$

论文里通常还会除以组内标准差；本教学实现保留最简单的均值中心化，便于和原版教程逐行对照。

## 2. 本地训练循环的五个步骤

```text
GSM8K 题目
  ↓ tokenizer.apply_chat_template
prompt tokens
  ↓ model.generate(group_size)
一组 completion + 采样时 scores
  ↓ log_softmax(scores) 取实际 token 的 logprob
old_logprobs + 规则 reward
  ↓ reward - mean(reward)
group advantages
  ↓ 当前模型 forward + policy loss + backward
LoRA 参数更新
```

### 2.1 Prompt 和 reward

`local_gpu.py` 通过 `scripts/local_gpu_common.py` 的 AST 加载器复用原版 `03-grpo.py` 的纯函数，不导入 `pytrio` / `swanlab` / `datasets`：

- `extract_boxed`：取最后一个 `\boxed{...}`。
- `normalize_answer`：去掉逗号和末尾句点。
- `extract_gsm8k_answer`：从 GSM8K 答案的 `####` 后取标准答案。
- `grade_answer`：完全一致 reward = 1，否则 = 0。

这是 RLVR（Reinforcement Learning with Verifiable Rewards）最舒服的地方：不需要 reward model，只要能写规则打分函数，就能先把训练闭环跑起来。

### 2.2 采样必须来自当前策略

原版 PyTRIO 会 `save_weights_and_get_sampling_client()`，明确把当前 LoRA 权重冻结成 old policy。本地版没有独立 sampler，但语义不变：

```python
with torch.no_grad():
    output = model.generate(
        do_sample=True,
        num_return_sequences=group_size,
        output_scores=True,
    )
```

`output.scores` 是每一步未归一化的 logits；对实际生成的 token 做 `log_softmax`，得到 `old_logprobs`。训练时再让当前模型 forward 同一段 `prompt + completion`，得到 `current_logprobs`。`ratio = exp(current - old)` 正是第 01 章的概率比例。

### 2.3 组内 advantage 与退化组

`common.group_advantages(rewards)` 做均值中心化。比如一组 4 条回答里有 3 条正确：

```text
reward     = [1, 1, 1, 0]
advantage  = [0.25, 0.25, 0.25, -0.75]
```

如果一组回答全对或全错，所有 advantage 都是 0，没有相对训练信号。代码会记 `degenerate=1.0` 并跳过 `backward`，这和原版 `frac_degenerate` 的含义一致。

### 2.4 从 completion 到本地 loss

原版把 `target_tokens`、`logprobs`、`advantages` 对齐后构造成 PyTRIO `Datum`。本地版直接拼张量：

```text
input_ids      = prompt + completion
labels         = input_ids[1:]
completion 区间 = [prompt_length - 1, prompt_length - 1 + completion_length)
```

`scripts/local_gpu_trainer.py` 的 `sequence_policy_loss()` 只在这段 completion 区间取 `current_logprobs`，再按命令行选三种 loss：

| Loss | 本地公式 |
| --- | --- |
| `importance_sampling` | `-(ratio * advantage).mean()` |
| `ppo` | `-min(ratio * advantage, clipped_ratio * advantage).mean()` |
| `cispo` | `-(detach(clipped_ratio) * current_logprob * advantage).mean()` |

然后就是普通 PyTorch 训练：

```python
loss.backward()
clip_grad_norm_(trainable_parameters, 1.0)
optimizer.step()
```

这里没有远程 `forward_backward()` / `optim_step()`，读者可以直接看到梯度从 loss 回到 LoRA 参数。

### 2.5 QLoRA 与显存边界

模型以 4-bit NF4 加载：

```python
BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4")
prepare_model_for_kbit_training(...)
get_peft_model(..., LoraConfig(target_modules="all-linear"))
```

目标环境是 24GB VRAM 的魔搭免费 NVIDIA 实例。第一版不做 8-bit/full fine-tuning 降级；bitsandbytes、CUDA 或显存检查失败会明确报错，避免“看起来跑完了，其实偷偷换了实现”。

## 3. 看哪些产物

- `config.json`：模型、数据、profile、loss、steps、group size。
- `samples.jsonl`：每步的问题、标准答案、模型回答、reward、advantage。
- `history.json` / `chapter-02.jsonl`：reward、degenerate、训练样本数、平均生成长度、loss。
- `summary.json`：最终摘要、adapter 路径、torch/GPU 信息。
- `experiment-records/state.json`：章节状态是否 `completed` / `failed`。
- `experiment-records/latest.md`：全局章节报告。

一个健康的 smoke 结果不一定 reward 上升；2 步、2 个回答的配置主要用于确认下载、生成、reward、loss、backward、保存和同步都走通。`degenerate=1.0` 也是合法观察：说明这一组全对或全错，暂时没有相对信号。

## 4. 和原版 PyTRIO 的对照

| 变量 / 动作 | 原版 PyTRIO | 本地 GPU 版 |
| --- | --- | --- |
| 当前策略 | 保存 sampler weights | 直接用训练中的 PEFT model `generate` |
| 旧策略 logprob | sampler 返回 | `output_scores` → `log_softmax` |
| 训练输入 | `trio.Datum` | `input_ids + attention_mask` |
| forward/backward | 远端 `forward_backward()` | 本地 `model(**inputs)` + `loss.backward()` |
| 优化器 | 远端 `optim_step(AdamParams)` | 本地 `AdamW` |
| 日志 | SwanLab | `/mnt/workspace` + Git 范围内 JSONL/Markdown |
| 权重 | 远端 sampler weights | `runs/02-grpo/<run>/adapter/` |

原版 `03-grpo.py` 仍然是很好的对照材料：它展示了把 datum/loss 交给云端 Train API 后，实验代码可以有多短。本地版则把这些被封装掉的步骤摊开，适合学习训练循环本身。

## 5. 已知简化

- 只做均值中心化，不除以组内标准差。
- 没有 value model、reference KL、异步 rollout、vLLM/SGLang 或分布式训练。
- 第一版逐题采样；`full` 的 `batch_size` 记录在配置中，但不把多题塞进同一个 forward，以降低 24GB 实例 OOM 风险。
- 不调用外部搜索、裁判或教师 API；预算账本只记录配置和密钥存在性。
- 结果用于学习训练闭环，不承诺复现论文指标。

下一章把同样的 GRPO 闭环推进到图文输入：图片如何进入 processor、视觉 token 如何和文本 token 共用坐标系、以及 HF GeoQA 不可达时如何如实记录 `data_unavailable`。
