# 第 01 章：本地 GPU 上的强化学习损失函数

这一章不下载 4B 模型，也不调用 PyTRIO。我们用一段很小的 PyTorch 张量实验，在魔搭单卡 GPU 上亲手把 `importance_sampling`、`ppo`、`cispo` 三个 loss 的公式、数值和梯度跑出来。目的是先建立直觉：RL 的 loss 不是“答案对不对”的交叉熵，而是把一条采样轨迹里每个 token 的相对好坏转成可求导目标。

本章脚本：[`local_gpu.py`](./local_gpu.py)
统一启动说明：[`MODELSCOPE_LOCAL_GPU.md`](../../MODELSCOPE_LOCAL_GPU.md)
建议阅读顺序：`01 Loss Functions → 02 GRPO → 03 Vision GRPO`

## 0. 先运行，再看公式

在仓库根目录执行：

```bash
bash scripts/run_modelscope.sh chapter 01
```

默认产物写到两个位置：

```text
/mnt/workspace/agentic-rl-lab/
├── runs/01-loss-functions/<run_id>/losses.json   # 持久化完整结果
├── logs/chapter-01.log                          # stdout/stderr
└── metrics/chapter-01.jsonl                     # 指标

experiments/loss-functions/results/               # 自动同步到 Git 范围
└── 01-loss-functions/<run_id>/losses.json
```

在服务器执行 `git add experiments/loss-functions/results experiment-records && git commit && git push` 后，本地 `git pull` 就能看到这次实验记录。大模型权重、缓存、`.env` 都不会进入 Git；服务器释放前仍可用 `bash scripts/export_results.sh` 下载完整归档。

`losses.json` 里有四个部分：

- `inputs`：三个 token 的旧策略 logprob、当前 logprob 和 advantage。
- `per_token_losses`：三个 loss 对每个 token 的数值。
- `gradients`：每个 loss 对 `target_logprobs` 的梯度。
- `scalar_examples`：公共模块 `scripts/local_gpu_common.py` 中同一公式的标量结果。

把数值和梯度一起看，比只看公式更容易理解 PPO 的 clip 与 CISPO 的 `detach` 到底改变了什么。

## 1. 三个变量

监督学习通常直接比较模型输出和标签；强化学习里，模型先自己生成一段回答，之后才拿到 reward。为了把整段回答的好坏分摊到 token 上，最少需要三个量：

- `advantage`：这条轨迹或这个 token 相对 baseline 是好还是坏。
- `sampling_logprobs`：采样时，旧策略有多喜欢这个 token。
- `target_logprobs`：训练时，当前模型现在有多喜欢这个 token。

概率比例写成：

```python
prob_ratio = torch.exp(target_logprobs - sampling_logprobs)
```

优化器默认最小化 loss，所以策略梯度的目标前面要加负号。我们真正想最大化的是“概率比例 × advantage”。

## 2. Importance Sampling

Importance Sampling 最直接：当前模型比采样模型更喜欢一个 token 多少，就按这个比例放大它的 advantage。

```math
\mathcal{L}_{\text{IS}}(\theta) = -\mathbb{E}\left[\frac{p_\theta(x)}{q(x)}A(x)\right]
```

对应 `local_gpu.py`：

```python
ratio = torch.exp(current_logprob - old_logprob)
loss = -(ratio * advantage)
```

如果 `advantage > 0`，训练会推高这个 token 的概率；如果 `advantage < 0`，训练会压低它。没有 clip，优点是梯度形态最清楚，缺点是一次更新可能走得很远。

## 3. PPO

PPO 解决的是“ratio 太大时别猛冲”。它对目标做 clip，让高到离谱的概率比例不再继续提供额外奖励：

```math
\mathcal{L}_{\text{PPO}}(\theta) =
-\mathbb{E}\left[\min\left(rA, \operatorname{clip}(r, 1-\epsilon, 1+\epsilon)A\right)\right]
```

本章用 `clip_high = 1.2`，所以代码里是：

```python
ppo = -torch.minimum(ratio * advantage, clip_high * advantage)
```

注意这不是“随便取小”，而是在做保守更新。看 `gradients` 时，被 clip 的 token 梯度会明显不同，这正是 PPO 的限速效果。

## 4. CISPO

PPO clip 的是整个 objective；CISPO clip 的是 ratio，然后把这个 ratio 当成固定权重乘到 `logprob * advantage` 上：

```math
\mathcal{L}_{\text{CISPO}}(\theta) =
-\mathbb{E}\left[\operatorname{sg}(\operatorname{clip}(r)) \cdot \log p_\theta(x) \cdot A(x)\right]
```

关键是 `sg` / `detach`：clipped ratio 不再参与梯度回传，真正被优化的是 `target_logprob`。

```python
clipped_ratio = torch.clamp(ratio, 1.0 / clip_high, clip_high).detach()
loss = -(clipped_ratio * current_logprob * advantage)
```

所以 CISPO 更像“给学习力度挂一个固定砝码”，而不是直接把某些 token 的目标剪没。

## 5. 一句话比较

| Loss | 人话比喻 | 核心动作 | 看 `gradients` 时观察什么 |
| --- | --- | --- | --- |
| Importance Sampling | 按功劳发奖金 | `ratio * advantage` | 正负 advantage 直接改变梯度方向 |
| PPO | 奖金有预算上限 | `min(unclipped, clipped)` | 被 clip 的位置不再继续放大更新 |
| CISPO | 固定砝码调节力度 | `detach(clipped_ratio) * logprob * advantage` | ratio 只改权重，梯度仍来自 logprob |

```text
IS:    有用就提高概率，没用就降低概率；ratio 负责修正采样偏差。
PPO:   还是这么做，但 ratio 不能太离谱，防止策略更新过猛。
CISPO: ratio 也会被 clip，但 clip 后只当固定权重，真正优化 logprob。
```

## 6. 从这一章到 GRPO

这一章的 `advantage` 是手写的教学张量。下一章会让它来自真实模型：同一道 GSM8K 题目采样一组回答，用规则 reward 打分，再用组内平均分作为 baseline。三个变量仍然是 `old_logprob`、`current_logprob`、`advantage`，只是它们由本地 `model.generate` 与模型 forward 产生。

原版 PyTRIO 教程仍保留在本仓库历史和其它章节中，可作为“云端 Train API 如何接管 forward/backward”的对照；本 fork 的主线是从本地 GPU 代码读懂训练循环本身。
