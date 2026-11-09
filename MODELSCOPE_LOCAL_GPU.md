# 魔搭 GPU：本地训练循环改造说明

这个 fork 以**单卡魔搭 GPU 本地实现**为主线：01–03 章 `readme.md` 已改成对应本地 GPU 教程，同目录 `local_gpu.py` 把采样、旧策略 logprob、组内 advantage、loss 和 QLoRA 更新放在魔搭免费 NVIDIA 实例的同一张 GPU 上。原版 PyTRIO 代码仍保留作对照；04–16 如实标记为 `not_ported`，不会伪造成功结果。

## 一句话启动

在服务器拉下仓库后，从仓库根目录执行：

```bash
git clone https://github.com/kuangxiaoc/agentic-rl-lab_cc.git
cd agentic-rl-lab_cc
bash scripts/bootstrap_modelscope.sh
bash scripts/run_modelscope.sh
```

默认按阅读顺序运行 `1-3`，默认 profile 是 `smoke`。第 07 章（DAPO）与第 08 章（GSPO）已有本地 GPU 版，用 `bash scripts/run_modelscope.sh chapter 07`（或 `08`）单章运行。第 04–06 章（PyTrio-Jev / General OPD / Medical OPD）尚未移植，会如实标记 `not_ported`。首次运行会检查并安装缺失的训练库，但**不会**覆盖镜像自带的 GPU 版 `torch`；请永远不要在服务器执行 `uv sync --locked`，因为原仓库的 `uv.lock` 指向 CPU torch。若镜像没有 CUDA `torch`，bootstrap 会直接失败并提示选择 CUDA 13.0 / torch 2.13 环境。

所有产物统一写入 `/mnt/workspace/agentic-rl-lab/`：

```text
/mnt/workspace/agentic-rl-lab/
├── state.json              # 16 个章节的持久状态，原子写入
├── api_budget.json         # 外部 API 免费额度与 ¥50 硬上限
├── environment.json        # Python/GPU/依赖/密钥是否存在的脱敏快照
├── study_notes.md          # 中文学习顺序与新旧代码对照
├── runs/                   # 每次实验的配置、样本、指标、LoRA adapter
├── logs/                   # 每章 stdout/stderr
├── metrics/                # JSONL 指标
├── reports/latest.md       # 章节状态报告
├── cache/                  # 模型/数据缓存；归档时排除大权重
└── exports/                # 可下载的 tar.gz
```

服务器释放前执行：

```bash
bash scripts/export_results.sh
```

下载输出的 `exports/agentic-rl-lab-*.tar.gz`。归档包含 `results/`（状态、日志、指标、adapter、报告、学习笔记）和 `code/`（本 fork 可读源码与原版章节），排除 `.env`、`.git`、完整模型缓存与超过 100 MiB 的文件。模型大权重不打包；LoRA adapter 会打包，适合本地继续看结构和小文件。

另外，runner 与 `export_results.sh` 都会把轻量实验记录同步到 Git 范围：

```text
experiments/loss-functions/results/
experiments/grpo/results/
experiments/vision-grpo/results/
experiment-records/
```

服务器跑完后执行 `git add experiments/*/results experiment-records && git commit && git push`，本地 `git pull` 就能看到 JSON/JSONL、日志、样本、学习笔记和报告。`.safetensors`、`.bin`、`.pt`、`.parquet`、模型缓存、`.env` 和单个超过 2 MiB 的产物不会复制到 Git；完整大产物仍由 `export_results.sh` 打包下载。

## 按顺序执行章节

```bash
# 一键：默认 1-3
bash scripts/run_modelscope.sh

# 单章跑 DAPO（第 07 章）
bash scripts/run_modelscope.sh chapter 07

# 显式一键：4-6 会如实标记 not_ported
RUN_CHAPTERS=4-6 bash scripts/run_modelscope.sh

# 单章：按原阅读顺序，一个一个执行
bash scripts/run_modelscope.sh chapter 01
bash scripts/run_modelscope.sh chapter 02
bash scripts/run_modelscope.sh chapter 03

# 已完成章节会自动跳过；需要重跑时加 --force
bash scripts/run_modelscope.sh chapter 02 --force

# 只检查环境、不下载模型
bash scripts/run_modelscope.sh doctor
```

`RUN_PROFILE=smoke` 默认配置是 `steps=2, batch_size=1, group_size=2, max_new_tokens=128`；`RUN_PROFILE=full` 是 `steps=10, batch_size=2, group_size=4, max_new_tokens=512`。02/03 当前按章节逐题采样；`full` 的 batch_size 是持久配置记录，第一版不会把多题塞进同一个 forward，以降低 24GB 显存实例的 OOM 风险。

## 01–03 分别做什么

- **01 Loss Functions**：不下载 4B 模型，用 PyTorch 张量在本地 GPU 上计算 `importance_sampling`、`ppo`、`cispo` 的 loss 和梯度。结果写到 `runs/01-loss-functions/<run>/losses.json`，与 `experiments/loss-functions/readme.md` 对照阅读。
- **02 GRPO**：从魔搭下载 `Qwen/Qwen3-4B-Instruct-2507` 与 `AI-ModelScope/gsm8k` 的 `main/train-00000-of-00001.parquet`，用 4-bit NF4 QLoRA 做本地 `model.generate`、规则 reward、组内 advantage 和真实 `loss.backward()`。原版的 `extract_boxed` / `normalize_answer` / `extract_gsm8k_answer` / `grade_answer` 通过 AST 纯函数复用，不导入 PyTRIO。可加 `--loss-fn ppo` 或 `cispo`；默认 `importance_sampling`。
- **03 Vision GRPO**：先对 HF `hz2475/geoQA` 做网络/parquet/schema 预检；只有拿到带图片的 train 行才下载 `Qwen/Qwen3-VL-4B-Instruct` 并启动多模态 QLoRA。若 HF 不可达或字段不符，状态写成 `data_unavailable` 并继续打包报告，**不会**拿 GSM8K 文本数据冒充视觉实验。原版的 `extract_choice` / `format_question` 同样复用。

三个章节都会在 `state.json` 记录 `running → completed / data_unavailable / failed`。单章失败会停止后续章节；按需可用 `bash scripts/run_modelscope.sh run 1-3 --continue-on-error`（runner 参数）让报告继续生成。

## 外部 API 与免费额度

第一版 01–03 **不发起任何外部搜索、裁判或教师 API 调用**；只预留配置、预算账本和密钥存在性检查，方便以后学习 Search-R1 / Tournament-GRPO / OPD：

- 硬上限：`MAX_EXTERNAL_COST_CNY=50`、`ALLOW_PAID_OVERAGE=0`、每个模型 `FREE_TOKEN_LIMIT=1000000`。
- 默认低成本候选：`deepseek-v4.1-flash`（裁判）、`qwen3.8-flash`（教师候选），关闭 thinking；不默认 `qwen3.8-max-0902`、`glm-5.3`、`kimi-k3` 或大参数模型。
- 搜索：Tavily 每月 1000 次免费优先；SerpAPI 每月 250 次作为备选。未来调用前必须通过 `record_external_usage()` 的额度检查。

你截图里的百炼免费额度是**每个模型/快照各自约 100 万输入+输出 token、有效期 90 天、仅北京地域实时推理**。跑通本地单卡 01–03 通常不用这些 API；1M token 对以后少量裁判/教师实验是够入门的，但密集 OPD 会很快消耗。务必在百炼控制台开启「免费额度用完即停」，否则免费额度耗尽后可能自动转为按量付费。密钥请放在 `/mnt/workspace/agentic-rl-lab/.env`，不要把真实值写进仓库；可参考 `scripts/.env.example`。

注意：OPD 原版依赖 teacher 对学生轨迹的 token logprob，普通百炼文本生成 API 不能等价替代；未来迁移需要单独设计，不会把“低价教师模型”当作已经复现 OPD。

## 与原版学习路径的对照

按 `experiments/README.md` 的顺序读：前三章 `readme.md` 现在直接以本地 GPU 为主线，讲清楚同目录 `local_gpu.py` 里 tensor/generate/processor/forward/backward 的每一步；原版 `03-grpo.py`、`train.py` 仍保留作 PyTRIO datum/loss 边界对照。原版 PyTRIO 依赖与 `pyproject.toml` 保持不变，便于把云端跑过的实验记录带到本地慢慢看。

框架调用依据：

- ModelScope `snapshot_download` / 数据集下载：[模型下载](https://www.modelscope.cn/docs/models/download)、[数据集下载](https://www.modelscope.cn/docs/datasets/download)
- Qwen3‑VL `AutoProcessor` + `apply_chat_template`：[Transformers Qwen3-VL](https://huggingface.co/docs/transformers/main/model_doc/qwen3_vl)
- 4-bit QLoRA：`BitsAndBytesConfig(load_in_4bit=True)` + `prepare_model_for_kbit_training` + `get_peft_model`：[bitsandbytes quickstart](https://huggingface.co/docs/bitsandbytes/main/quickstart)
- 采样配置：`do_sample`、`num_return_sequences`、`temperature`、`top_p`：[Transformers generation](https://huggingface.co/docs/transformers/main_classes/text_generation)

## 已知边界

- 目标环境固定为 Ubuntu 22.04、Python 3.12、CUDA 13.0.3、GPU torch 2.13、24GB 显存；不支持 AMD/ROCm，不提供 CPU 或低精度静默降级。
- 4-bit bitsandbytes 失败会明确报错，不会偷偷换 8-bit/full fine-tuning。
- 这是教学用最小 GRPO 循环：没有 value model、reference KL、异步 rollout、vLLM/SGLang 或分布式训练；03 若数据不可用会明确记录。
- 本地只做标准库测试、语法检查和 shell 语法检查；真实模型下载/首次 GPU smoke 必须在服务器完成。
