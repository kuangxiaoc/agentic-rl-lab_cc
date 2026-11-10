# 魔搭 GPU：本地训练循环改造说明

这个 fork 以**单卡魔搭 GPU 本地实现**为主线：01–09 章已有对应本地 GPU 教程或附录，同目录 `local_gpu.py` 把采样、旧策略 logprob、advantage、loss 和 QLoRA 更新放在魔搭免费 NVIDIA 实例的同一张 GPU 上。原版 PyTRIO 代码仍保留作对照；10–16 如实标记为 `not_ported`，不会伪造成功结果。

## 一句话启动

在服务器拉下仓库后，从仓库根目录执行：

```bash
git clone https://github.com/kuangxiaoc/agentic-rl-lab_cc.git
cd agentic-rl-lab_cc
bash scripts/bootstrap_modelscope.sh
bash scripts/run_modelscope.sh
```

默认按阅读顺序运行 `1-3`，默认 profile 是 `smoke`；要一次跑完当前已移植的章节用 `RUN_CHAPTERS=1-9 bash scripts/run_modelscope.sh`。第 04 章（PyTrio-Jev）使用 Qwen3-4B 做单次 forward 的 masked 决策 SFT；第 05 章（General OPD）使用 Qwen3-1.7B student + Qwen3-4B teacher；第 06 章（Medical OPD）使用同一组 Qwen3 student/teacher，并增加 Base-anchor 或 IDT 模式。第 07 章（DAPO）与第 08 章（GSPO）也已本地化。第 09 章（OPSD）用同一个 Qwen3-4B 的 Student/Teacher 双 prompt 做 sampled-token reverse KL。10–16 仍会如实标记 `not_ported`。首次运行会检查并安装缺失的训练库，但**不会**覆盖镜像自带的 GPU 版 `torch`；请永远不要在服务器执行 `uv sync --locked`，因为原仓库的 `uv.lock` 指向 CPU torch。若镜像没有 CUDA `torch`，bootstrap 会直接失败并提示选择 CUDA 13.0 / torch 2.13 环境。

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

# 显式一键：04–06 本地 OPD / 决策章节
RUN_CHAPTERS=4-6 bash scripts/run_modelscope.sh

# 当前已移植的完整顺序
RUN_CHAPTERS=1-8 bash scripts/run_modelscope.sh

# 单章：按原阅读顺序，一个一个执行
bash scripts/run_modelscope.sh chapter 01
bash scripts/run_modelscope.sh chapter 02
bash scripts/run_modelscope.sh chapter 03

# 已完成章节会自动跳过；需要重跑时加 --force
bash scripts/run_modelscope.sh chapter 02 --force

# 只检查环境、不下载模型
bash scripts/run_modelscope.sh doctor
```

`RUN_PROFILE=smoke` 默认配置是 `steps=2, batch_size=1, group_size=2, max_new_tokens=128`；`RUN_PROFILE=full` 是 `steps=10, batch_size=2, group_size=4, max_new_tokens=512`。02/03 当前按章节逐题采样；`full` 的 batch_size 是持久配置记录，第一版不会把多题塞进同一个 forward，以降低 24GB 显存实例的 OOM 风险。04–06 也读取同一个 `RUN_PROFILE`；它们各自还提供 `--steps` / `--sft-steps` 等更细的覆盖参数。

## 01–09 分别做什么

- **01 Loss Functions**：不下载 4B 模型，用 PyTorch 张量在本地 GPU 上计算 `importance_sampling`、`ppo`、`cispo` 的 loss 和梯度。结果写到 `runs/01-loss-functions/<run>/losses.json`，与 `experiments/loss-functions/readme.md` 对照阅读。
- **02 GRPO**：从魔搭下载 `Qwen/Qwen3-4B-Instruct-2507` 与 `AI-ModelScope/gsm8k` 的 `main/train-00000-of-00001.parquet`，用 4-bit NF4 QLoRA 做本地 `model.generate`、规则 reward、组内 advantage 和真实 `loss.backward()`。原版的 `extract_boxed` / `normalize_answer` / `extract_gsm8k_answer` / `grade_answer` 通过 AST 纯函数复用，不导入 PyTRIO。可加 `--loss-fn ppo` 或 `cispo`；默认 `importance_sampling`。
- **03 Vision GRPO**：先对 HF `hz2475/geoQA` 做网络/parquet/schema 预检；只有拿到带图片的 train 行才下载 `Qwen/Qwen3-VL-4B-Instruct` 并启动多模态 QLoRA。若 HF 不可达或字段不符，状态写成 `data_unavailable` 并继续打包报告，**不会**拿 GSM8K 文本数据冒充视觉实验。原版的 `extract_choice` / `format_question` 同样复用。
- **04 PyTrio-Jev**：从 ModelScope 下载 `LocalLLaMA/typed-decisions`，用 Qwen3-4B-Instruct-2507 做 marker-only masked SFT。每个字段只在 `decision` marker 前一个位置监督答案符号，评测时在候选符号上做受限 softmax。`ToolACE` 不是必需依赖，避免公开镜像不可用时整章失败。
- **05 General OPD**：从 ModelScope 下载 DeepMath-103K 的一个 parquet 分片，Qwen3-1.7B student 先采样，Qwen3-4B-Instruct-2507 teacher 对同一条 token 轨迹计算 logprob，本地计算 `reverse_kl = student - teacher`，再用 PPO/IS 更新 student。
- **06 Medical OPD**：本地 Medical SFT 训练 Qwen3-4B Medical Teacher；Qwen3-1.7B Student 先做 Medical OPD，再用原始 Qwen3-1.7B 做 Base-anchor OPD。`--mode idt` 可切换为 Medical/C-Eval 双 Teacher 交替训练。MedQA-zh 与 C-Eval 都在同一张卡上本地评测。

04–06 也会在 `state.json` 记录 `running → completed / data_unavailable / failed`。单章失败会停止后续章节；按需可用 `bash scripts/run_modelscope.sh run 4-6 --continue-on-error`（runner 参数）让报告继续生成。每章的 `readme.md` 末尾都有本地 GPU 运行附录，说明模型对、命令、产物路径和与原版规模的边界。

## 外部 API 与免费额度

当前 01–09 **不发起任何外部搜索、裁判或教师 API 调用**；只预留配置、预算账本和密钥存在性检查，方便以后学习 Search-R1 / Tournament-GRPO 等章节：

- 硬上限：`MAX_EXTERNAL_COST_CNY=50`、`ALLOW_PAID_OVERAGE=0`、每个模型 `FREE_TOKEN_LIMIT=1000000`。
- 默认低成本候选：`deepseek-v4.1-flash`（裁判）、`qwen3.8-flash`（教师候选），关闭 thinking；不默认 `qwen3.8-max-0902`、`glm-5.3`、`kimi-k3` 或大参数模型。
- 搜索：Tavily 每月 1000 次免费优先；SerpAPI 每月 250 次作为备选。未来调用前必须通过 `record_external_usage()` 的额度检查。

你截图里的百炼免费额度是**每个模型/快照各自约 100 万输入+输出 token、有效期 90 天、仅北京地域实时推理**。跑通本地单卡 01–09 通常不用这些 API；1M token 对以后少量裁判实验是够入门的，但普通聊天 API 不能替代 OPD Teacher 的逐 token logprob。务必在百炼控制台开启「免费额度用完即停」，否则免费额度耗尽后可能自动转为按量付费。密钥请放在 `/mnt/workspace/agentic-rl-lab/.env`，不要把真实值写进仓库；可参考 `scripts/.env.example`。

注意：OPD 依赖 teacher 对学生轨迹的 token logprob，普通百炼文本生成 API 不能等价替代；本 fork 的 05/06 使用本地 4-bit teacher forward 计算 logprob，不把“低价教师 API”当作已经复现 OPD。

## 与原版学习路径的对照

按 `experiments/README.md` 的顺序读：01–03 章的 `readme.md` 以本地 GPU 为主线，04–06 章末尾新增本地运行附录，讲清楚同目录 `local_gpu.py` 里 tensor/generate/processor/forward/backward 的每一步；原版 `03-grpo.py`、`train.py` 等仍保留作 PyTRIO datum/loss 边界对照。原版 PyTRIO 依赖与 `pyproject.toml` 保持不变，便于把云端跑过的实验记录带到本地慢慢看。

框架调用依据：

- ModelScope `snapshot_download` / 数据集下载：[模型下载](https://www.modelscope.cn/docs/models/download)、[数据集下载](https://www.modelscope.cn/docs/datasets/download)
- Qwen3‑VL `AutoProcessor` + `apply_chat_template`：[Transformers Qwen3-VL](https://huggingface.co/docs/transformers/main/model_doc/qwen3_vl)
- 4-bit QLoRA：`BitsAndBytesConfig(load_in_4bit=True)` + `prepare_model_for_kbit_training` + `get_peft_model`：[bitsandbytes quickstart](https://huggingface.co/docs/bitsandbytes/main/quickstart)
- 采样配置：`do_sample`、`num_return_sequences`、`temperature`、`top_p`：[Transformers generation](https://huggingface.co/docs/transformers/main_classes/text_generation)

## 已知边界

- 目标环境固定为 Ubuntu 22.04、Python 3.12、CUDA 13.0.3、GPU torch 2.13、24GB 显存；不支持 AMD/ROCm，不提供 CPU 或低精度静默降级。
- 4-bit bitsandbytes 失败会明确报错，不会偷偷换 8-bit/full fine-tuning。
- 这是教学用最小本地训练循环：没有 value model、reference KL、异步 rollout、vLLM/SGLang 或分布式训练；03 若数据不可用会明确记录。
- 05/06 使用 1.7B student + 4B teacher，是为了在 A10 上稳定跑通机制，不等同于原版 4B student / 27B teacher 的论文规模；不要把本地 smoke 的准确率直接和原版表格比较。
- 04 的 ToolACE 可选数据默认不下载；如果后续需要完整五项 Jevbench 对照，再单独补 ToolACE 镜像或本地 JSONL。
- 本地只做标准库测试、语法检查和 shell 语法检查；真实模型下载/首次 GPU smoke 必须在服务器完成。
