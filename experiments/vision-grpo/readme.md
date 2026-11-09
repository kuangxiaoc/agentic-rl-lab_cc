# 第 03 章：本地 GPU 上的 Vision GRPO / GeoQA

这一章把第 02 章的文本 GRPO 闭环推进到多模态：模型先看一张几何图，再读取题目和四个选项，对同一道题采样一组回答，用 `\boxed{A-D}` 选项奖励做组内比较，并在本地 4-bit QLoRA 模型上完成参数更新。

本章脚本：[`local_gpu.py`](./local_gpu.py)
统一启动说明：[`MODELSCOPE_LOCAL_GPU.md`](../../MODELSCOPE_LOCAL_GPU.md)
原版 PyTRIO 参考：[`train.py`](./train.py)
建议阅读顺序：`01 Loss Functions → 02 GRPO → 03 Vision GRPO`

## 0. 先预检数据，再训练

在仓库根目录执行：

```bash
bash scripts/run_modelscope.sh chapter 03
```

执行顺序刻意和纯文本章节不同：

1. 从 HF `hz2475/geoQA` 预检 `data/*.parquet`。
2. 过滤 `original_split == "train"`，确认每行有图片和四个选项。
3. 预检通过后，才从魔搭下载 `Qwen/Qwen3-VL-4B-Instruct`。
4. 检查 CUDA / 24GB 显存级别 GPU，再加载 4-bit QLoRA 模型。
5. 启动本地多模态采样、reward、advantage 和 backward。

如果 HF 不可达、parquet 不存在、字段不符合预期或图片无法解码，章节状态会写成 `data_unavailable`，并继续生成报告与归档。代码**不会**用 GSM8K 文本数据冒充视觉实验，也不会伪造成功结果。

## 1. 产物在哪里，怎么带回本地

持久化完整结果：

```text
/mnt/workspace/agentic-rl-lab/
├── runs/03-vision-grpo/<run_id>/
│   ├── data_preflight.json
│   ├── config.json
│   ├── samples.jsonl
│   ├── history.json
│   ├── summary.json
│   ├── data_unavailable.txt     # 仅数据预检失败时
│   └── adapter/                # LoRA 权重，留在持久目录
├── logs/chapter-03.log
└── metrics/chapter-03.jsonl
```

runner 结束后自动同步到 Git 范围：

```text
experiments/vision-grpo/results/
├── 03-vision-grpo/<run_id>/*.json / *.jsonl / *.txt
├── chapter-03.log
└── chapter-03.jsonl

experiment-records/
├── state.json                 # 包含 completed / data_unavailable / failed
├── environment.json
├── latest.md
└── study_notes.md
```

服务器执行：

```bash
git add experiments/vision-grpo/results experiment-records
git commit -m "record: local GPU vision GRPO"
git push
```

本地 `git pull` 就能看到数据预检结果、题目 ID、gold choice、模型回答、预测选项、reward、advantage、loss 和最终状态。模型缓存、图片 parquet、`.env`、adapter `.safetensors` 不进入 Git；完整归档仍用：

```bash
bash scripts/export_results.sh
```

`RUN_PROFILE=smoke` 默认 `steps=2, group_size=2, max_new_tokens=128`；`RUN_PROFILE=full` 为 `steps=10, group_size=4, max_new_tokens=512`。第一版逐题处理，不并发多题 rollout，以降低 24GB 显存实例的 OOM 风险。

## 2. 一张图片怎样进入本地模型

原版 PyTRIO 把渲染后的图文 prompt 拆成 `EncodedTextChunk + ImageChunk + EncodedTextChunk`，并手动计算 `expected_tokens`，以保证本地 prompt 长度和远端 `input_tokens` 一致。

本地版使用 Transformers 的 `AutoProcessor.apply_chat_template()`，让它一次性返回：

```python
inputs = processor.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_dict=True,
    return_tensors="pt",
)
```

返回的 `input_ids` 里已经包含视觉占位 token；`pixel_values` / `image_grid_thw` 等字段保存图像张量和网格信息。`scripts/local_gpu_trainer.py` 会把这些张量移到 CUDA，并把浮点视觉输入转为 BF16。

消息结构和原版逻辑一致：

```python
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": format_question(subject, choices)},
        ],
    }
]
```

`format_question()`、`extract_choice()` 通过 AST 加载器从原版 `train.py` 复用，不导入 `pytrio` / `swanlab` / `PIL` 等训练时无关依赖。GeoQA parquet 里的图片可能是 PIL 对象，也可能是 `{"bytes": ...}`；`as_rgb_image()` 会把两者统一转成 RGB。

### 为什么不再手动检查 `input_tokens` 对齐

本地 processor 同时产出文本 token 和视觉张量，模型 forward 直接消费同一批 `inputs`。没有“本地构造 chunk → 远端重新展开”的坐标系转换，因此不需要原版的 `response.input_tokens == len(prompt)` 检查。取而代之的是更直接的边界：训练时把 `input_ids` 与 `pixel_values` 一起送进同一个本地模型，completion logits 只从 prompt 之后切出。

## 3. Vision GRPO 的闭环

```text
GeoQA 图文题目
  ↓ AutoProcessor.apply_chat_template
input_ids + pixel_values / image_grid_thw
  ↓ model.generate(group_size)
一组 completion + 采样 scores
  ↓ extract_choice(最后一个 \boxed{A-D})
0/1 rule reward
  ↓ reward - mean(reward)
group advantages
  ↓ 图文输入 + completion 重新 forward
policy loss + backward + AdamW
```

### 3.1 Reward

正确选项由 `label` 转成 `A/B/C/D`。模型回答中最后一个 `\boxed{A}` 到 `\boxed{D}` 与标签一致时 reward = 1，否则为 0。没有可解析 boxed 选项也记 0，并体现在 `format_rate` 指标里。

### 3.2 Group-relative advantage

设一组 `G` 条回答的 reward 为 `r_i`：

$$
A_i = r_i - \frac{1}{G}\sum_{j=1}^{G}r_j
$$

例如 4 条里 3 条正确：

```text
reward    = [1, 1, 1, 0]
advantage = [0.25, 0.25, 0.25, -0.75]
```

全对或全错时 advantage 全为 0，代码记为 `degenerate`，跳过 backward。这不是失败，而是这一组没有相对训练信号。

### 3.3 图文样本如何做 backward

采样得到 completion 后，代码把 `prompt + completion` 作为新的 `input_ids`，并复用同一条样本的视觉张量：

```python
extra_inputs = {
    key: value[:1]
    for key, value in inputs.items()
    if key not in {"input_ids", "attention_mask"}
}
```

`sequence_policy_loss()` 把这些视觉字段和 `input_ids` 一起传给本地模型，只在 completion 对应位置取 `current_logprobs`。因此图片不是监督标签，也不参与 reward；它只提供上下文，让模型在同一图文条件下学习哪种选项轨迹更值得生成。

## 4. 三个 loss 仍然来自第 01 章

`--loss-fn` 可选：

| Loss | 本地公式 |
| --- | --- |
| `importance_sampling` | `-(ratio * advantage).mean()` |
| `ppo` | `-min(ratio * advantage, clipped_ratio * advantage).mean()` |
| `cispo` | `-(detach(clipped_ratio) * current_logprob * advantage).mean()` |

其中：

```python
ratio = exp(current_logprobs - old_logprobs)
```

`old_logprobs` 来自 `model.generate(output_scores=True)`，`current_logprobs` 来自训练 forward。视觉输入改变的只是 prompt 的构造方式；GRPO 的核心变量没有变。

## 5. 看哪些记录

- `data_preflight.json`：HF GeoQA 预检行数。
- `config.json`：模型、数据、profile、loss、采样参数。
- `samples.jsonl`：每步的 `id`、gold choice、预测、回答、reward、advantage。
- `history.json` / `chapter-03.jsonl`：reward、`format_rate`、`degenerate`、训练样本数、平均生成长度、loss。
- `summary.json`：最终摘要、adapter 路径、torch/GPU 信息。
- `data_unavailable.txt`：数据不可用的具体原因。
- `experiment-records/latest.md`：16 个章节状态；09–16 暂未移植，04–08 已有本地代码但需要实际运行后才变成 `completed`。

smoke 的目标不是刷准确率，而是确认：图片能进 processor、生成 scores 能转 old logprob、boxed 选项能打分、图文 forward 能 backward、状态和记录能同步到 Git。

## 6. 和原版 PyTRIO 的对照

| 问题 | 原版 PyTRIO | 本地 GPU 版 |
| --- | --- | --- |
| 图片表示 | 手动编码成 `ImageChunk` | `AutoProcessor` 产出 `pixel_values` 等张量 |
| 长度对齐 | 本地估计 vs 远端 `input_tokens` | 同一 processor / 同一模型，无需跨服务对齐 |
| 采样 | 远端 sampler | 本地 PEFT model `generate` |
| 训练输入 | 多模态 `trio.Datum` | `input_ids + attention_mask + 视觉张量` |
| 梯度更新 | 远端 forward/backward | 本地 `loss.backward()` |
| 日志与权重 | SwanLab + 远端 checkpoint | `/mnt/workspace` + Git 范围内 JSON/Markdown |

原版 `train.py` 仍然适合学习“远程服务要求哪些显式字段”；本地版则适合学习 Transformers 多模态 processor 已经帮你做了什么。

## 7. 已知边界

- 目标仅限 NVIDIA CUDA、BF16、24GB 显存级别；不支持 AMD/ROCm，不提供 CPU 静默降级。
- 4-bit bitsandbytes 失败会明确报错，不会退化成 8-bit 或全量微调。
- HF GeoQA 数据不可用会写 `data_unavailable`，绝不替换成纯文本数据。
- 没有 value model、reference KL、异步 rollout、vLLM/SGLang、分布式训练或独立评测脚本。
- 原版 100-step 结果和图表属于 PyTRIO 云端实验记录，不代表本本地 GPU smoke 的效果。
- 第一版不调用外部搜索、裁判或教师 API。

完成本地章节后，建议把 `experiment-records/latest.md`、各章 `results/` 和 `exports/*.tar.gz` 一起带回本地：前者适合用 Git 版本化地回看每次实验，后者保存完整 adapter 和运行环境记录。
