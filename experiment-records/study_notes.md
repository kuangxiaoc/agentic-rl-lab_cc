# 本地 GPU 学习笔记

原版借助 PyTRIO 把采样、forward/backward 和优化器更新放在云端；本 fork 保留原版章节，
在每个原章节目录新增 `local_gpu.py`，把同一套 reward / advantage / loss 放到 ModelScope
单卡 GPU 上亲手实现。按 `experiments/README.md` 的阅读顺序看：

1. `experiments/loss-functions/readme.md` → `local_gpu.py`：用张量复现 IS / PPO / CISPO。
2. `experiments/grpo/readme.md` → `03-grpo.py` → `local_gpu.py`：对比 PyTRIO Datum 与
   本地 `model.generate`、旧策略 logprob、组内 advantage、PEFT 梯度更新。
3. `experiments/vision-grpo/readme.md` → `train.py` → `local_gpu.py`：观察图片如何进入
   processor、旧版 chunk 对齐如何变成本地 attention mask 与 pixel values。
4. 04-16 暂时标记 `not_ported`，不会伪造成功结果；外部搜索/裁判/教师 API 先只记录预算。

每次运行的配置、stdout/stderr、metrics、LoRA adapter、`state.json` 和环境快照都在本目录。
服务器释放前执行 `bash scripts/export_results.sh`，下载 `exports/*.tar.gz` 即可离线学习。
