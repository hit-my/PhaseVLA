# PhaseVLA

PhaseVLA is an independent research repository built on top of [OpenPI](https://github.com/Physical-Intelligence/openpi) and the $\pi_{0.5}$ vision-language-action (VLA) model. It studies whether explicit recurrent memory can help a frozen VLA infer task progress from prior observations and avoid repeating completed actions.

The current implementation adds a FutureMamba memory plugin to the PyTorch $\pi_{0.5}$ pipeline:

- **Recurrent memory:** a Mamba-2 backend receives the final VLM representation and previous memory state, then produces the next state.
- **Progress Expert:** a lightweight expert reads the memory token and predicts an early denoising direction.
- **Early-step intervention:** the plugin intervenes in the first handoff steps, while the frozen Action Expert retains responsibility for later action refinement.
- **Frozen base model:** the $\pi_{0.5}$ VLM and Action Expert remain frozen; only the FutureMamba plugin is trained.

This is the standalone PhaseVLA research codebase, not a GitHub fork. OpenPI is retained as the upstream implementation and attribution base.

## PhaseVLA Progress

### Implemented

- Pure PyTorch FutureMamba model, Mamba-2 memory state contract, and Progress Expert.
- RoboMME execution-sample dataset indexing with episode-boundary protection and burn-in handling.
- Training, checkpoint save/restore, frozen-base verification, and policy-server inference paths.
- Deterministic RoboMME experiment matrix, strict result aggregation, memory-swap intervention evaluation, and profiling contracts.
- Independent structural and loss configurations for `progress_depth`, handoff ratio, and `flow-only` training.

### Current evaluation snapshot

The current formal RoboMME scope contains three Counting tasks: `BinFill`, `PickXtimes`, and `SwingXtimes`. The checkpoint sweep below uses one independently trained task-specific model per task (`handoff_ratio=0.4`, batch size 4, train seed 42). Every cell is a validation evaluation over episode IDs 0–49 with evaluation seed 7; aggregates describe the three task-specific models and are not a single shared-model evaluation.

| Checkpoint | BinFill | PickXtimes | SwingXtimes | Aggregate |
| ---: | ---: | ---: | ---: | ---: |
| 500 | 17/50 (34%) | 13/50 (26%) | 20/50 (40%) | 50/150 (33.33%) |
| 1000 | 14/50 (28%) | 16/50 (32%) | 22/50 (44%) | 52/150 (34.67%) |
| 1500 | 14/50 (28%) | 15/50 (30%) | 24/50 (48%) | 53/150 (35.33%) |
| 2000 | 15/50 (30%) | 13/50 (26%) | 27/50 (54%) | 55/150 (36.67%) |
| 2500 | 15/50 (30%) | 12/50 (24%) | 27/50 (54%) | 54/150 (36.00%) |

Best observed checkpoints are BinFill step 500 (34%), PickXtimes step 1000 (32%), and SwingXtimes step 2000/2500 (54%). The historical shared-model step-5000 snapshot (`17/15/21 = 53/150`) is retained as prior evidence but is not mixed with this task-specific sweep.

The PI0.5 LIBERO-Mem baseline checkpoint at step 10000 was also evaluated formally on all 10 tasks, 20 rollouts per task (200 total), with rollout seed 10001. It achieved `2/200` successes (1.00%, Wilson 95% CI 0.27–3.57%). The complete audited record is `/data/libero_mem_baseline/formal_10k_libero_mem_audited_summary.json`; the raw 200-rollout file has SHA256 `787e290aca11d1317bcc6106e8b9d69d5a76011b5d3f5f505704e15dd44f8375`.

### Current training status (2026-08-23)

- LIBERO-Mem PI0.5 is running at step 41000/50000 on four A100 80 GB GPUs (`batch_size=64`, `fsdp_devices=4`), resumed from the durable step-10000 checkpoint. Cross-topology Orbax restore now explicitly reshards FSDP2 checkpoint arrays onto the FSDP4 mesh. The next durable checkpoint target is step 45000.
- Four exploratory FutureMamba runs were intentionally stopped on 2026-08-22. Their final logged steps were 1100/5000 for `handoff=0.4 + stride8 + batch4`, and 390/2500, 430/2500, and 540/2500 for the `handoff=0.7 + batch16` BinFill, PickXtimes, and SwingXtimes runs. These partial runs are not reported as final evaluation results.
- No `train_futuremamba_pytorch.py` process is currently active. LIBERO-Mem training remains active and was not interrupted by stopping the FutureMamba jobs.

### Ablation, data-pipeline, and mechanism evidence

- Full-episode variable-length RoboMME training, independent online/training query strides, sparse train-query masks, and observation-only memory updates are implemented.
- Conditioning-cache production and consumption now support cached final hidden states, variable-length prefix masks, and Action Expert KV tensors with provenance checks.
- Flow-only training supports a deterministic terminal-loss monitor that is logged but never added to the optimized objective.
- Registered configurations cover light Progress Expert depth, handoff ratios, stride-8 online memory updates, batch-size variants, memory depth/width, and six-layer Progress Expert placement.
- Real CUDA one-step smoke tests completed for `progress_depth=4`, `progress_depth=9`, `flow-only`, `handoff_ratio=0.4`, and `handoff_ratio=0.0`.
- A memory-backend mechanism experiment covered GRU, LSTM, FrameStack, NoMemory, and Mamba-2; sequence outputs were finite and reset/snapshot/restore behavior was checked.
- The latest 18 modified Python files compile successfully. Ten targeted test files covering training, checkpointing, conditioning-cache integrity, model/config, policy/server, episode collation, and RoboMME datasets pass: **143 passed in 11.42s**. This is a targeted regression run, not the project-wide suite.

### Known limitations

- Full multi-seed formal evaluation has not been completed; all current RoboMME checkpoint results use train seed 42 and evaluation seed 7.
- The handoff=0.7/batch16 and handoff=0.4/stride8/batch4 runs were stopped before completion and must not be treated as final ablations.
- The audited LIBERO-Mem step-10000 baseline result is only 1.00%; later checkpoints still require the same 10-task × 20-rollout protocol before any training-progress claim can be made.
- Formal FLOPs, peak-memory, and end-to-end episode-timing profiles with a single provenance-bound artifact set remain incomplete.
- Real-robot validation is not included in the current evidence snapshot.
- The RoboMME mirror configured in `.gitmodules` is private and requires access when initializing submodules.

Runtime evaluation artifacts and large checkpoints remain outside Git. Persisted result paths and hashes are recorded above and in `/data/phasevla/PhaseVLA-results.md`.

## 服务器迁移交接手册

本节用于将 PhaseVLA / FutureMamba 迁移到另一台远程服务器。请按顺序执行。命令中的 `/path/to/PhaseVLA`、`/path/to/data`、`/path/to/runs` 和 `<HF_TOKEN>` 需要替换为新服务器上的实际路径，或通过交互式登录提供。不要把 Token、SSH 私钥或 W&B API Key 写入 Git。

### 1. 当前项目身份与迁移边界

| 项目 | 当前值 |
| --- | --- |
| 主仓库 | `git@github.com:hit-my/PhaseVLA.git`（私有） |
| 交接前 README 基线提交 | `648ee738b273697acd8c4bce77596e7b3a9c6de4` |
| 已发布迁移代码分支 | `main` |
| 本地开发分支 | `docs/futuremamba-pytorch-migration-plan` |
| 主仓库主线 | `main` |
| RoboMME 策略子模块 | `third_party/robomme_policy_learning`，commit `9e627481745a7cf8a5725a140a00e8befedff7b6` |
| RoboMME benchmark 子模块 | `third_party/robomme_policy_learning/third_party/robomme_benchmark`，commit `856bc3a189d4172f3f47dbee4424d585f8d78db3` |
| RoboMME 策略来源 commit | `ecf086c3be7c2223167d9bb2f6ef1f0a6e24353b` |
| Mamba 来源 commit | `77069de5cdb55cbe98b670889c80df211e031039` |
| 主评测平台 | RoboMME 官方 ManiSkill/SAPIEN 闭环 |
| 当前主后端 | Mamba-2 |

当前旧工作区曾出现 `third_party/robomme_policy_learning` 子模块工作树有未跟踪内层 benchmark 的状态。新服务器不要复制这个脏状态；以主仓库提交、`.gitmodules` 和递归子模块固定 commit 为准重新初始化。

### 2. 新服务器硬件与软件基线

当前已验证基线：Ubuntu 22.04、Python 3.11.15、`uv 0.10.12`、NVIDIA RTX 5090（32,607 MiB）、驱动 `580.173.02`、PyTorch `2.9.1+cu128`、CUDA runtime `12.8`、Triton `3.5.1`，以及官方 `mamba-ssm` commit `77069de5cdb55cbe98b670889c80df211e031039`。

当前 shell 默认 `nvcc` 曾为 CUDA 11.5，不能用于 RTX 5090 / Mamba 构建。新服务器构建时必须显式使用 CUDA 12.8：

```bash
export CUDA_HOME=/usr/local/cuda-12.8
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
python3.11 --version
nvcc --version
nvidia-smi
uv --version
```

新服务器 GPU 可以不同，但正式对比前必须记录 GPU、驱动、CUDA、PyTorch、Triton 和 Mamba commit。不同软件栈的结果不要直接与当前 RTX 5090 结果合并。

### 3. 克隆私有仓库与固定子模块

先确认新服务器上的 SSH key 或 GitHub CLI 凭据同时有权访问 `hit-my/PhaseVLA` 和私有仓库 `hit-my/PhaseVLA-robomme-policy-learning`：

```bash
mkdir -p /path/to
cd /path/to
git clone --branch main \
  --recurse-submodules git@github.com:hit-my/PhaseVLA.git PhaseVLA
cd PhaseVLA

git remote -v
git rev-parse HEAD
git submodule sync --recursive
git submodule update --init --recursive
git submodule status --recursive
```

预期递归子模块包含以下 commit：

```text
third_party/robomme_policy_learning              @ 9e627481745a7cf8a5725a140a00e8befedff7b6
third_party/robomme_policy_learning/third_party/robomme_benchmark
                                                   @ 856bc3a189d4172f3f47dbee4424d585f8d78db3
```

当前 FutureMamba 迁移代码和本交接手册已发布在 `main`。本地开发分支名为 `docs/futuremamba-pytorch-migration-plan`，不需要从远程单独检出。不要用 `git pull` 覆盖包含实验产物的目录。

### 4. 创建两个隔离环境

策略环境使用仓库内 `environments/futuremamba`；RoboMME 仿真环境与策略环境隔离。策略环境固定 Python `<3.12`、Torch `2.9.1`、Triton `3.5.1` 和 Mamba commit：

```bash
cd /path/to/PhaseVLA
export CUDA_HOME=/usr/local/cuda-12.8
uv venv --python 3.11 environments/futuremamba/.venv
uv sync --project environments/futuremamba
```

如需重新构建 Mamba：

```bash
CUDA_HOME=/usr/local/cuda-12.8 \
  uv sync --project environments/futuremamba --reinstall-package mamba-ssm
```

验证策略环境：

```bash
environments/futuremamba/.venv/bin/python - <<'PY'
import torch
import triton
import mamba_ssm

print("torch", torch.__version__, "cuda", torch.version.cuda)
print("cuda_available", torch.cuda.is_available())
print("triton", triton.__version__)
print("mamba_ssm", mamba_ssm.__file__)
if torch.cuda.is_available():
    print("gpu", torch.cuda.get_device_name(0))
PY
```

创建 RoboMME 仿真环境：

```bash
micromamba create -n robomme python=3.11 -y
micromamba run -n robomme pip install -r \
  third_party/robomme_policy_learning/examples/robomme/requirements.txt
micromamba run -n robomme pip install -e \
  third_party/robomme_policy_learning/third_party/robomme_benchmark
micromamba run -n robomme pip install -e packages/openpi-client
micromamba run -n robomme python \
  third_party/robomme_policy_learning/examples/robomme/simple_test.py
```

`simple_test.py` 应以退出码 0 完成至少一个仿真 step。若服务器没有显示器，按 RoboMME 文档配置 EGL/Vulkan headless runtime，不要修改任务定义、评测器或成功判定。

### 5. 迁移数据、checkpoint 与结果

推荐在新服务器保持以下目录布局：

```text
/path/to/PhaseVLA/
├── data/robomme_data_h5/
├── runs/ckpts/pi05_baseline/
├── runs/ckpts/pi05_baseline_pytorch/79999/
├── checkpoints/futuremamba_robomme_mamba2/
└── runs/evaluation/
```

需要从旧服务器或受控对象存储传输：RoboMME 原始数据及 checksum、JAX `pi05_baseline` checkpoint、PyTorch 基座、FutureMamba 插件 checkpoint，以及 `progress.json`、`log.json`、视频和结果 JSON。插件 checkpoint 至少保留 `metadata.json`、`plugin.safetensors`、`optimizer.pt`、`scheduler.pt` 和 `rng_state.pt`。

当前旧服务器产物大小约为：`/tmp/futuremamba_delivery` 9.1 MB、完整验证 checkpoint 712 MB、各 CUDA smoke checkpoint 约 665 MB–1.4 GB。建议使用 `rsync` 断点传输：

```bash
rsync -avP --partial /source/data/robomme_data_h5/ \
  /path/to/PhaseVLA/data/robomme_data_h5/
rsync -avP --partial /source/runs/ckpts/ \
  /path/to/PhaseVLA/runs/ckpts/
rsync -avP --partial /source/checkpoints/ \
  /path/to/PhaseVLA/checkpoints/
```

迁移后核对：

```bash
du -sh data/robomme_data_h5 runs/ckpts checkpoints
find runs/ckpts/pi05_baseline_pytorch/79999 -maxdepth 2 -type f -print
find checkpoints -name metadata.json -o -name plugin.safetensors
```

不要把 `/tmp` 路径写入长期配置；新服务器应使用持久化磁盘。

### 6. 下载缺失的官方数据与 checkpoint

如果没有从旧服务器传输大文件，可在新服务器重新下载。需要 Hugging Face 权限时，使用交互式 `huggingface-cli login`，不要把 Token 写入 shell 历史：

```bash
mkdir -p data runs/ckpts
git clone https://huggingface.co/datasets/Yinpei/robomme_data_h5 data/robomme_data_h5
uv run third_party/robomme_policy_learning/scripts/tarxz_h5.py decompress \
  --input_dir data/robomme_data_h5 --jobs 16 --remove_archive

git clone https://huggingface.co/Yinpei/pi05_baseline runs/ckpts/pi05_baseline
uv run third_party/robomme_policy_learning/scripts/unzip_ckpt.py \
  runs/ckpts/pi05_baseline
```

确认后期 checkpoint 至少包含 `params/` 和 `assets/robomme/norm_stats.json`：

```bash
find runs/ckpts/pi05_baseline -maxdepth 5 \
  \( -type d -name params -o -type f -name norm_stats.json \) -print
```

RoboMME 正式基座不是 LIBERO checkpoint。若实际后期 checkpoint 不是 `79999`，同步更新配置、checkpoint metadata 和评测 mapping。

### 7. 将 JAX 基座转换为 PyTorch

只有在 `params/`、RoboMME assets 和 checksum 齐全后执行：

```bash
PYTHONPATH=src environments/futuremamba/.venv/bin/python \
  examples/convert_jax_model_to_pytorch.py \
  --checkpoint_dir runs/ckpts/pi05_baseline/pi05_baseline/79999 \
  --config_name pi05_robomme_pytorch \
  --output_path runs/ckpts/pi05_baseline_pytorch/79999 \
  --precision float32
```

转换结果应包含 `model.safetensors`、`config.json`、`conversion_manifest.json` 和复制后的 `assets/`。如果转换报告 `params` 或 RoboMME norm stats 缺失，停止，不要用不完整目录训练。

### 8. 新服务器最小验收

先不启动长时间训练，执行代码、依赖、Mamba 和 checkpoint smoke：

```bash
cd /path/to/PhaseVLA
export CUDA_HOME=/usr/local/cuda-12.8
export PYTHONPATH="$PWD/src"
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1

environments/futuremamba/.venv/bin/pytest -q \
  src/openpi/models_pytorch/futuremamba_config_test.py \
  src/openpi/models_pytorch/futuremamba_test.py \
  src/openpi/policies/futuremamba_policy_test.py \
  src/openpi/training/futuremamba_checkpoint_test.py \
  scripts/train_futuremamba_pytorch_test.py

environments/futuremamba/.venv/bin/python -m compileall -q \
  src/openpi/models_pytorch src/openpi/policies src/openpi/training scripts
```

然后运行 Mamba-2 相关测试；若新 GPU 不支持当前 kernel，记录失败原因，不要伪造通过：

```bash
environments/futuremamba/.venv/bin/pytest -q \
  src/openpi/models_pytorch/mamba_memory_test.py
```

RoboMME `simple_test.py`、策略测试和 CUDA smoke 均有记录后，才启动 WebSocket 服务和 episode 评测。

### 9. 训练 FutureMamba 与断点续训

默认 RoboMME 配置为 `futuremamba_robomme_mamba2`，主设置包括 `progress_depth=6`、`handoff_ratio=0.2`、`num_denoise_steps=10`、`action_horizon=20`、在线执行 horizon 16 和 `memory_backend="mamba2"`。训练脚本参数如下：

```bash
CUDA_VISIBLE_DEVICES=0 \
  environments/futuremamba/.venv/bin/python \
  scripts/train_futuremamba_pytorch.py futuremamba_robomme_mamba2 \
  --seed 42 \
  --episode-data-dir /path/to/robomme_episode_samples \
  --num-train-steps 5000 \
  --save-interval 5000 \
  --log-interval 100 \
  --pytorch-training-precision bfloat16 \
  --wandb-enabled false \
  --checkpoint-root checkpoints/futuremamba_robomme_mamba2/seed42
```

断点续训必须保持同一配置、基座 checksum、RoboMME dataset checksum 和 memory state schema：

```bash
CUDA_VISIBLE_DEVICES=0 \
  environments/futuremamba/.venv/bin/python \
  scripts/train_futuremamba_pytorch.py futuremamba_robomme_mamba2 \
  --seed 42 \
  --episode-data-dir /path/to/robomme_episode_samples \
  --num-train-steps 10000 \
  --checkpoint-root checkpoints/futuremamba_robomme_mamba2/seed42 \
  --resume
```

训练脚本只允许 `futuremamba.*` 参数可训练，并在 checkpoint metadata 中记录 Torch、CUDA、GPU、Mamba、数据和基座身份。不要覆盖已有 checkpoint；重新开始时使用新的 `--checkpoint-root`。

### 10. 启动策略服务与运行三任务单 episode

FutureMamba bundle 必须同时包含 `plugin.safetensors` 和 `metadata.json`。metadata 还会严格校验冻结 PyTorch 基座的 `model.safetensors`、assets checksum、RoboMME commit、Mamba commit 和 memory state schema。保持仓库根目录作为当前工作目录，并保持 `runs/ckpts/pi05_baseline_pytorch/79999` 的相对路径：

```bash
cd /path/to/PhaseVLA
export CUDA_HOME=/usr/local/cuda-12.8
export PYTHONPATH="$PWD/src"

CUDA_VISIBLE_DEVICES=0 \
  environments/futuremamba/.venv/bin/python \
  scripts/serve_policy.py policy:checkpoint \
  --policy.config=futuremamba_robomme_mamba2 \
  --policy.dir=runs/ckpts/futuremamba_robomme_mamba2/pickxtimes_seed42/5000 \
  --port=8000
```

另开终端，三任务均应至少完成 1 个 validation episode。下面先运行 `PickXtimes`；将 `--task-name` 改为 `BinFill` 或 `SwingXtimes` 可复用同一命令：

```bash
cd /path/to/PhaseVLA
micromamba run -n robomme python \
  scripts/run_robomme_single_episode.py \
  --task-name PickXtimes \
  --episode-id 0 \
  --split validation \
  --dataset val \
  --host 127.0.0.1 \
  --port 8000 \
  --use-history true \
  --policy-name futuremamba_mamba2 \
  --model-seed 42 \
  --model-ckpt-id 5000 \
  --save-dir runs/evaluation/futuremamba_mamba2_ckpt5000
```

单 episode 结果必须保存视频、`progress.json`、`log.json` 和服务端日志；仅服务启动成功不算评测通过。

### 11. 生成三任务确定性评测矩阵

先准备 checkpoint mapping JSON。每个 checkpoint 条目必须包含路径、配置名、后端、训练 seed、checkpoint ID 和 provenance；不要用自动扫描到的未知 checkpoint 直接生成论文结果。当前完整评测唯一使用 `counting` 阶段：

```bash
environments/futuremamba/.venv/bin/python \
  scripts/run_robomme_experiment_matrix.py \
  --checkpoint-mapping /path/to/checkpoint_mapping.json \
  --output runs/evaluation/manifest_counting.json \
  --stage counting \
  --port 8000 \
  --eval-seed 7
```

`counting` 固定生成 `BinFill`、`PickXtimes`、`SwingXtimes` 三个 validation 任务，每个任务 50 个 episode，共 150 个 episode。`minimal` 仅用于迁移后的快速 smoke。评测记录必须保留 task、episode、train seed、eval seed、checkpoint、config、server 和 provenance 字段。

### 12. 当前服务器结果、训练状态和待办

截至 2026-08-23，已核验的正式结果包括：

- FutureMamba 三个任务特定模型的 step 500/1000/1500/2000/2500 checkpoint 均完成 50-episode validation 测评；完整表见 README 顶部。
- 任务最佳结果为 `BinFill 17/50`（step 500）、`PickXtimes 16/50`（step 1000）、`SwingXtimes 27/50`（step 2000/2500）。
- LIBERO-Mem PI0.5 step 10000 已完成 10 tasks × 20 rollouts，结果 `2/200`（1.00%，Wilson 95% CI `[0.27%, 3.57%]`），200 条记录覆盖完整且 identity violations 为 0。
- LIBERO-Mem 当前运行在四张 A100 80 GB 上，最新核验 step 41000/50000；FutureMamba 的四个探索性训练进程已按要求停止，不再占用 GPU。

最新代码增量包括：全 episode 变长序列、独立 `train_query_stride`、稀疏训练 query mask、conditioning cache 及变长 prefix padding、memory-only 在线更新、terminal monitor、stride-8 在线评测参数、扩展实验配置，以及 FSDP2→FSDP4 checkpoint 显式重分片恢复。18 个变更 Python 文件已通过 `py_compile`；10 个定向测试文件共 `143 passed`。

后续优先级：

1. 等待 LIBERO-Mem step 45000/50000 checkpoint 完整落盘，并按相同 10×20 正式协议测评，禁止用训练 loss 代替成功率；
2. 对 FutureMamba 正式结果补齐多 seed 重复实验；
3. 在同一硬件/软件栈完成 FLOPs、训练/推理峰值显存和 episode timing profile；
4. 最后进行真机验证。其他 RoboMME 任务仍不纳入当前统计或结论。

主要持久化产物：

```text
/data/phasevla/formal_task_specific_handoff_0p4_batch4/
/data/libero_mem_baseline/formal_10k_libero_mem_rollouts.jsonl
/data/libero_mem_baseline/formal_10k_libero_mem_audited_summary.json
/data/phasevla/PhaseVLA-results.md
```

### 13. 故障排查

| 现象 | 处理 |
| --- | --- |
| `Permission denied (publickey)` | 检查新服务器 SSH agent、GitHub key 和两个私有仓库权限；不要改用无权限的公共 URL。 |
| `mamba_ssm` 编译失败或找不到 `sm_120` | 设置 `CUDA_HOME=/usr/local/cuda-12.8`，确认 `nvcc --version`，清理失败 wheel 后重装。 |
| `torch.cuda.is_available()` 为 `False` | 检查 NVIDIA 驱动、容器 GPU 透传、Torch CUDA wheel 和 `CUDA_VISIBLE_DEVICES`。 |
| 子模块显示 `-`、`?` 或 commit 不匹配 | 执行 `git submodule sync --recursive && git submodule update --init --recursive`，再核对固定 commit。 |
| checkpoint metadata 不匹配 | 不要强行加载；核对 base checksum、assets checksum、RoboMME commit、dataset checksum、Mamba commit 和 state schema。 |
| `norm_stats.json` 缺失 | 检查 PyTorch 基座 assets 和 `RoboMMEDataConfig.assets_dir`，不要使用其他任务的 norm stats。 |
| RoboMME 无显示/Vulkan 报错 | 按 RoboMME 文档配置 EGL/Vulkan headless runtime；不要修改 evaluator 或任务语义。 |
| 显存不足 | 先降低 smoke batch/window 或关闭非必要服务；不要把改变模型结构的临时配置作为正式结果。 |
| W&B 登录失败 | 使用 `--wandb-enabled false` 做本地 smoke，正式训练前再配置 W&B 凭据。 |

### 14. 交接完成判据

- [ ] 能读取私有 `PhaseVLA` 主仓库和 RoboMME 子模块；
- [ ] 主仓库、递归子模块 commit 与本节记录一致；
- [ ] `CUDA_HOME` 指向 CUDA 12.8，Torch/CUDA/Triton/Mamba 版本已记录；
- [ ] `futuremamba` 策略环境和 `robomme` 仿真环境均可导入；
- [ ] RoboMME 官方 `simple_test.py` 退出码为 0；
- [ ] FutureMamba 定向测试和 Mamba-2 smoke 已运行并记录输出；
- [ ] 基座、assets、episode 数据和插件 checkpoint 的 checksum 已保存；
- [ ] WebSocket policy server 能启动，三个 in-scope 任务各至少完成 1 个 episode；
- [ ] 评测日志、视频、manifest 和 provenance 已写入持久化目录；
- [ ] 新服务器所有凭据均通过安全凭据管理，不进入 README、shell history 或 Git。

## Upstream OpenPI models

PhaseVLA retains the upstream OpenPI model support and documentation below. OpenPI currently contains three model types:

- the [π₀ model](https://www.physicalintelligence.company/blog/pi0), a flow-based VLA;
- the [π₀-FAST model](https://www.physicalintelligence.company/research/fast), an autoregressive VLA based on the FAST action tokenizer;
- the [π₀.₅ model](https://www.physicalintelligence.company/blog/pi05), an upgraded version of π₀ with better open-world generalization.

For upstream installation, conversion, inference, and fine-tuning instructions, see the corresponding sections below.

## Updates

- [Aug 2026] PhaseVLA: added the PyTorch FutureMamba memory plugin, RoboMME evaluation pipeline, ablation contracts, and current evaluation snapshot.
- [Aug 2026] Added full-episode/sparse-stride training, conditioning-cache integration, memory-only online updates, terminal monitoring, expanded RoboMME variants, and audited RoboMME/LIBERO-Mem progress records.
- [Sept 2025] OpenPI released PyTorch support.
- [Sept 2025] OpenPI released π₀.₅, an upgraded version of π₀ with better open-world generalization.
- [Sept 2025] OpenPI added an [improved idle filter](examples/droid/README_train.md#data-filtering) for DROID training.
- [Jun 2025] OpenPI added [instructions](examples/droid/README_train.md) for training VLAs on the full [DROID dataset](https://droid-dataset.github.io/).

## Requirements

To run the models in this repository, you will need an NVIDIA GPU with at least the following specifications. These estimations assume a single GPU, but you can also use multiple GPUs with model parallelism to reduce per-GPU memory requirements by configuring `fsdp_devices` in the training config. Please also note that the current training script does not yet support multi-node training.

| Mode               | Memory Required | Example GPU        |
| ------------------ | --------------- | ------------------ |
| Inference          | > 8 GB          | RTX 4090           |
| Fine-Tuning (LoRA) | > 22.5 GB       | RTX 4090           |
| Fine-Tuning (Full) | > 70 GB         | A100 (80GB) / H100 |

The repo has been tested with Ubuntu 22.04, we do not currently support other operating systems.

## Installation

When cloning PhaseVLA, initialize its submodules. The main repository and RoboMME mirror are private, so collaborators need access to both repositories:

```bash
git clone --recurse-submodules git@github.com:hit-my/PhaseVLA.git

# Or if you already cloned the repository:
git submodule update --init --recursive
```

We use [uv](https://docs.astral.sh/uv/) to manage Python dependencies. See the [uv installation instructions](https://docs.astral.sh/uv/getting-started/installation/) to set it up. Once uv is installed, run the following to set up the environment:

```bash
GIT_LFS_SKIP_SMUDGE=1 uv sync
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
```

NOTE: `GIT_LFS_SKIP_SMUDGE=1` is needed to pull LeRobot as a dependency.

**Docker**: As an alternative to uv installation, we provide instructions for installing openpi using Docker. If you encounter issues with your system setup, consider using Docker to simplify installation. See [Docker Setup](docs/docker.md) for more details.




## Model Checkpoints

### Base Models
We provide multiple base VLA model checkpoints. These checkpoints have been pre-trained on 10k+ hours of robot data, and can be used for fine-tuning.

| Model        | Use Case    | Description                                                                                                 | Checkpoint Path                                |
| ------------ | ----------- | ----------------------------------------------------------------------------------------------------------- | ---------------------------------------------- |
| $\pi_0$      | Fine-Tuning | Base [π₀ model](https://www.physicalintelligence.company/blog/pi0) for fine-tuning                | `gs://openpi-assets/checkpoints/pi0_base`      |
| $\pi_0$-FAST | Fine-Tuning | Base autoregressive [π₀-FAST model](https://www.physicalintelligence.company/research/fast) for fine-tuning | `gs://openpi-assets/checkpoints/pi0_fast_base` |
| $\pi_{0.5}$    | Fine-Tuning | Base [π₀.₅ model](https://www.physicalintelligence.company/blog/pi05) for fine-tuning    | `gs://openpi-assets/checkpoints/pi05_base`      |

### Fine-Tuned Models
We also provide "expert" checkpoints for various robot platforms and tasks. These models are fine-tuned from the base models above and intended to run directly on the target robot. These may or may not work on your particular robot. Since these checkpoints were fine-tuned on relatively small datasets collected with more widely available robots, such as ALOHA and the DROID Franka setup, they might not generalize to your particular setup, though we found some of these, especially the DROID checkpoint, to generalize quite broadly in practice.

| Model                    | Use Case    | Description                                                                                                                                                                                              | Checkpoint Path                                       |
| ------------------------ | ----------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------- |
| $\pi_0$-FAST-DROID       | Inference   | $\pi_0$-FAST model fine-tuned on the [DROID dataset](https://droid-dataset.github.io/): can perform a wide range of simple table-top manipulation tasks 0-shot in new scenes on the DROID robot platform | `gs://openpi-assets/checkpoints/pi0_fast_droid`       |
| $\pi_0$-DROID            | Fine-Tuning | $\pi_0$ model fine-tuned on the [DROID dataset](https://droid-dataset.github.io/): faster inference than $\pi_0$-FAST-DROID, but may not follow language commands as well                                | `gs://openpi-assets/checkpoints/pi0_droid`            |
| $\pi_0$-ALOHA-towel      | Inference   | $\pi_0$ model fine-tuned on internal [ALOHA](https://tonyzhaozh.github.io/aloha/) data: can fold diverse towels 0-shot on ALOHA robot platforms                                                          | `gs://openpi-assets/checkpoints/pi0_aloha_towel`      |
| $\pi_0$-ALOHA-tupperware | Inference   | $\pi_0$ model fine-tuned on internal [ALOHA](https://tonyzhaozh.github.io/aloha/) data: can unpack food from a tupperware container                                                                                                             | `gs://openpi-assets/checkpoints/pi0_aloha_tupperware` |
| $\pi_0$-ALOHA-pen-uncap  | Inference   | $\pi_0$ model fine-tuned on public [ALOHA](https://dit-policy.github.io/) data: can uncap a pen                                                                                                          | `gs://openpi-assets/checkpoints/pi0_aloha_pen_uncap`  |
| $\pi_{0.5}$-LIBERO      | Inference   | $\pi_{0.5}$ model fine-tuned for the [LIBERO](https://libero-project.github.io/datasets) benchmark: gets state-of-the-art performance (see [LIBERO README](examples/libero/README.md)) | `gs://openpi-assets/checkpoints/pi05_libero`      |
| $\pi_{0.5}$-DROID      | Inference / Fine-Tuning | $\pi_{0.5}$ model fine-tuned on the [DROID dataset](https://droid-dataset.github.io/) with [knowledge insulation](https://www.physicalintelligence.company/research/knowledge_insulation): fast inference and good language-following | `gs://openpi-assets/checkpoints/pi05_droid`      |


By default, checkpoints are automatically downloaded from `gs://openpi-assets` and are cached in `~/.cache/openpi` when needed. You can overwrite the download path by setting the `OPENPI_DATA_HOME` environment variable.




## Running Inference for a Pre-Trained Model

Our pre-trained model checkpoints can be run with a few lines of code (here our $\pi_0$-FAST-DROID model):
```python
from openpi.training import config as _config
from openpi.policies import policy_config
from openpi.shared import download

config = _config.get_config("pi05_droid")
checkpoint_dir = download.maybe_download("gs://openpi-assets/checkpoints/pi05_droid")

# Create a trained policy.
policy = policy_config.create_trained_policy(config, checkpoint_dir)

# Run inference on a dummy example.
example = {
    "observation/exterior_image_1_left": ...,
    "observation/wrist_image_left": ...,
    ...
    "prompt": "pick up the fork"
}
action_chunk = policy.infer(example)["actions"]
```
You can also test this out in the [example notebook](examples/inference.ipynb).

We provide detailed step-by-step examples for running inference of our pre-trained checkpoints on [DROID](examples/droid/README.md) and [ALOHA](examples/aloha_real/README.md) robots.

**Remote Inference**: We provide [examples and code](docs/remote_inference.md) for running inference of our models **remotely**: the model can run on a different server and stream actions to the robot via a websocket connection. This makes it easy to use more powerful GPUs off-robot and keep robot and policy environments separate.

**Test inference without a robot**: We provide a [script](examples/simple_client/README.md) for testing inference without a robot. This script will generate a random observation and run inference with the model. See [here](examples/simple_client/README.md) for more details.





## Fine-Tuning Base Models on Your Own Data

We will fine-tune the $\pi_{0.5}$ model on the [LIBERO dataset](https://libero-project.github.io/datasets) as a running example for how to fine-tune a base model on your own data. We will explain three steps:
1. Convert your data to a LeRobot dataset (which we use for training)
2. Defining training configs and running training
3. Spinning up a policy server and running inference

### 1. Convert your data to a LeRobot dataset

We provide a minimal example script for converting LIBERO data to a LeRobot dataset in [`examples/libero/convert_libero_data_to_lerobot.py`](examples/libero/convert_libero_data_to_lerobot.py). You can easily modify it to convert your own data! You can download the raw LIBERO dataset from [here](https://huggingface.co/datasets/openvla/modified_libero_rlds), and run the script with:

```bash
uv run examples/libero/convert_libero_data_to_lerobot.py --data_dir /path/to/your/libero/data
```

**Note:** If you just want to fine-tune on LIBERO, you can skip this step, because our LIBERO fine-tuning configs point to a pre-converted LIBERO dataset. This step is merely an example that you can adapt to your own data.

### 2. Defining training configs and running training

To fine-tune a base model on your own data, you need to define configs for data processing and training. We provide example configs with detailed comments for LIBERO below, which you can modify for your own dataset:

- [`LiberoInputs` and `LiberoOutputs`](src/openpi/policies/libero_policy.py): Defines the data mapping from the LIBERO environment to the model and vice versa. Will be used for both, training and inference.
- [`LeRobotLiberoDataConfig`](src/openpi/training/config.py): Defines how to process raw LIBERO data from LeRobot dataset for training.
- [`TrainConfig`](src/openpi/training/config.py): Defines fine-tuning hyperparameters, data config, and weight loader.

We provide example fine-tuning configs for [π₀](src/openpi/training/config.py), [π₀-FAST](src/openpi/training/config.py), and [π₀.₅](src/openpi/training/config.py) on LIBERO data.

Before we can run training, we need to compute the normalization statistics for the training data. Run the script below with the name of your training config:

```bash
uv run scripts/compute_norm_stats.py --config-name pi05_libero
```

Now we can kick off training with the following command (the `--overwrite` flag is used to overwrite existing checkpoints if you rerun fine-tuning with the same config):

```bash
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi05_libero --exp-name=my_experiment --overwrite
```

The command will log training progress to the console and save checkpoints to the `checkpoints` directory. You can also monitor training progress on the Weights & Biases dashboard. For maximally using the GPU memory, set `XLA_PYTHON_CLIENT_MEM_FRACTION=0.9` before running training -- this enables JAX to use up to 90% of the GPU memory (vs. the default of 75%).

**Note:** We provide functionality for *reloading* normalization statistics for state / action normalization from pre-training. This can be beneficial if you are fine-tuning to a new task on a robot that was part of our pre-training mixture. For more details on how to reload normalization statistics, see the [norm_stats.md](docs/norm_stats.md) file.

### 3. Spinning up a policy server and running inference

Once training is complete, we can run inference by spinning up a policy server and then querying it from a LIBERO evaluation script. Launching a model server is easy (we use the checkpoint for iteration 20,000 for this example, modify as needed):

```bash
uv run scripts/serve_policy.py policy:checkpoint --policy.config=pi05_libero --policy.dir=checkpoints/pi05_libero/my_experiment/20000
```

This will spin up a server that listens on port 8000 and waits for observations to be sent to it. We can then run an evaluation script (or robot runtime) that queries the server.

For running the LIBERO eval in particular, we provide (and recommend using) a Dockerized workflow that handles both the policy server and the evaluation script together. See the [LIBERO README](examples/libero/README.md) for more details.

If you want to embed a policy server call in your own robot runtime, we have a minimal example of how to do so in the [remote inference docs](docs/remote_inference.md).



### More Examples

We provide more examples for how to fine-tune and run inference with our models on the ALOHA platform in the following READMEs:
- [ALOHA Simulator](examples/aloha_sim)
- [ALOHA Real](examples/aloha_real)
- [UR5](examples/ur5)

## PyTorch Support

openpi now provides PyTorch implementations of π₀ and π₀.₅ models alongside the original JAX versions! The PyTorch implementation has been validated on the LIBERO benchmark (both inference and finetuning). A few features are currently not supported (this may change in the future):

- The π₀-FAST model
- Mixed precision training
- FSDP (fully-sharded data parallelism) training
- LoRA (low-rank adaptation) training
- EMA (exponential moving average) weights during training

### Setup
1. Make sure that you have the latest version of all dependencies installed: `uv sync`

2. Double check that you have transformers 4.53.2 installed: `uv pip show transformers`

3. Apply the transformers library patches:
   ```bash
   cp -r ./src/openpi/models_pytorch/transformers_replace/* .venv/lib/python3.11/site-packages/transformers/
   ```

This overwrites several files in the transformers library with necessary model changes: 1) supporting AdaRMS, 2) correctly controlling the precision of activations, and 3) allowing the KV cache to be used without being updated.

**WARNING**: With the default uv link mode (hardlink), this will permanently affect the transformers library in your uv cache, meaning the changes will survive reinstallations of transformers and could even propagate to other projects that use transformers. To fully undo this operation, you must run `uv cache clean transformers`.

### Converting JAX Models to PyTorch

To convert a JAX model checkpoint to PyTorch format:

```bash
uv run examples/convert_jax_model_to_pytorch.py \
    --checkpoint_dir /path/to/jax/checkpoint \
    --config_name <config name> \
    --output_path /path/to/converted/pytorch/checkpoint
```

### Running Inference with PyTorch

The PyTorch implementation uses the same API as the JAX version - you only need to change the checkpoint path to point to the converted PyTorch model:

```python
from openpi.training import config as _config
from openpi.policies import policy_config
from openpi.shared import download

config = _config.get_config("pi05_droid")
checkpoint_dir = "/path/to/converted/pytorch/checkpoint"

# Create a trained policy (automatically detects PyTorch format)
policy = policy_config.create_trained_policy(config, checkpoint_dir)

# Run inference (same API as JAX)
action_chunk = policy.infer(example)["actions"]
```

### Policy Server with PyTorch

The policy server works identically with PyTorch models - just point to the converted checkpoint directory:

```bash
uv run scripts/serve_policy.py policy:checkpoint \
    --policy.config=pi05_droid \
    --policy.dir=/path/to/converted/pytorch/checkpoint
```

### Finetuning with PyTorch

To finetune a model in PyTorch:

1. Convert the JAX base model to PyTorch format:
   ```bash
   uv run examples/convert_jax_model_to_pytorch.py \
       --config_name <config name> \
       --checkpoint_dir /path/to/jax/base/model \
       --output_path /path/to/pytorch/base/model
   ```

2. Specify the converted PyTorch model path in your config using `pytorch_weight_path`

3. Launch training using one of these modes:

```bash
# Single GPU training:
uv run scripts/train_pytorch.py <config_name> --exp_name <run_name> --save_interval <interval>

# Example:
uv run scripts/train_pytorch.py debug --exp_name pytorch_test
uv run scripts/train_pytorch.py debug --exp_name pytorch_test --resume  # Resume from latest checkpoint

# Multi-GPU training (single node):
uv run torchrun --standalone --nnodes=1 --nproc_per_node=<num_gpus> scripts/train_pytorch.py <config_name> --exp_name <run_name>

# Example:
uv run torchrun --standalone --nnodes=1 --nproc_per_node=2 scripts/train_pytorch.py pi0_aloha_sim --exp_name pytorch_ddp_test
uv run torchrun --standalone --nnodes=1 --nproc_per_node=2 scripts/train_pytorch.py pi0_aloha_sim --exp_name pytorch_ddp_test --resume

# Multi-Node Training:
uv run torchrun \
    --nnodes=<num_nodes> \
    --nproc_per_node=<gpus_per_node> \
    --node_rank=<rank_of_node> \
    --master_addr=<master_ip> \
    --master_port=<port> \
    scripts/train_pytorch.py <config_name> --exp_name=<run_name> --save_interval <interval>
```

### Precision Settings

JAX and PyTorch implementations handle precision as follows:

**JAX:**
1. Inference: most weights and computations in bfloat16, with a few computations in float32 for stability
2. Training: defaults to mixed precision: weights and gradients in float32, (most) activations and computations in bfloat16. You can change to full float32 training by setting `dtype` to float32 in the config.

**PyTorch:**
1. Inference: matches JAX -- most weights and computations in bfloat16, with a few weights converted to float32 for stability
2. Training: supports either full bfloat16 (default) or full float32. You can change it by setting `pytorch_training_precision` in the config. bfloat16 uses less memory but exhibits higher losses compared to float32. Mixed precision is not yet supported.

With torch.compile, inference speed is comparable between JAX and PyTorch.

## Troubleshooting

We will collect common issues and their solutions here. If you encounter an issue, please check here first. If you can't find a solution, please file an issue on the repo (see [here](CONTRIBUTING.md) for guidelines).

| Issue                                     | Resolution                                                                                                                                                                                   |
| ----------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `uv sync` fails with dependency conflicts | Try removing the virtual environment directory (`rm -rf .venv`) and running `uv sync` again. If issues persist, check that you have the latest version of `uv` installed (`uv self update`). |
| Training runs out of GPU memory           | Make sure you set `XLA_PYTHON_CLIENT_MEM_FRACTION=0.9` (or higher) before running training to allow JAX to use more GPU memory. You can also use `--fsdp-devices <n>` where `<n>` is your number of GPUs, to enable [fully-sharded data parallelism](https://engineering.fb.com/2021/07/15/open-source/fsdp/), which reduces memory usage in exchange for slower training (the amount of slowdown depends on your particular setup). If you are still running out of memory, you may want to consider disabling EMA.        |
| Policy server connection errors           | Check that the server is running and listening on the expected port. Verify network connectivity and firewall settings between client and server.                                            |
| Missing norm stats error when training    | Run `scripts/compute_norm_stats.py` with your config name before starting training.                                                                                                          |
| Dataset download fails                    | Check your internet connection. For HuggingFace datasets, ensure you're logged in (`huggingface-cli login`).                                                                                 |
| CUDA/GPU errors                           | Verify NVIDIA drivers are installed correctly. For Docker, ensure nvidia-container-toolkit is installed. Check GPU compatibility. You do NOT need CUDA libraries installed at a system level --- they will be installed via uv. You may even want to try *uninstalling* system CUDA libraries if you run into CUDA issues, since system libraries can sometimes cause conflicts. |
| Import errors when running examples       | Make sure you've installed all dependencies with `uv sync`. Some examples may have additional requirements listed in their READMEs.                    |
| Action dimensions mismatch                | Verify your data processing transforms match the expected input/output dimensions of your robot. Check the action space definitions in your policy classes.                                  |
| Diverging training loss                            | Check the `q01`, `q99`, and `std` values in `norm_stats.json` for your dataset. Certain dimensions that are rarely used can end up with very small `q01`, `q99`, or `std` values, leading to huge states and actions after normalization. You can manually adjust the norm stats as a workaround. |

## Latest LIBERO-Mem ten-task FutureMamba results

This section records the completed ten-task LIBERO-Mem FutureMamba experiment. Large runtime checkpoints, videos, and datasets remain outside Git; compact audited exports are committed under `results/`.

- Completion event: `2026-08-30 00:37:57 UTC`.
- Protocol: MuJoCo `3.2.2`, fixed stabilized initialization, 10 tasks, 20 episodes per task (episode IDs `0-19`), rollout seed `10001`, FutureMamba train seed `42`, `max_steps=600`, `replan_steps=20`, handoff ratio `0.4`.
- Conditioning cache: `961/961`; action cache: `961/961`; complete checkpoints: `60/60`; formal evaluations: `60/60`; audited rollouts/videos: `1200/1200` / `1200/1200`; audit violations: `0`.

### Overall success rates

| Method / checkpoint | Successes | Success rate | Wilson 95% CI | Delta vs baseline |
|---|---:|---:|---:|---:|
| baseline step-49999 | 162/200 | 81.0% | [75.00-85.83%] | - |
| FutureMamba step-500 | 166/200 | 83.0% | [77.18-87.57%] | +2.0 pp |
| FutureMamba step-1000 | 171/200 | 85.5% | [79.95-89.71%] | +4.5 pp |
| FutureMamba step-1500 | 168/200 | 84.0% | [78.29-88.43%] | +3.0 pp |
| FutureMamba step-2000 | 177/200 | 88.5% | [83.34-92.21%] | +7.5 pp |
| FutureMamba step-2500 | 168/200 | 84.0% | [78.29-88.43%] | +3.0 pp |
| FutureMamba step-3000 | 171/200 | 85.5% | [79.95-89.71%] | +4.5 pp |

The best common checkpoint is FutureMamba step-2000: `177/200 = 88.5%` (`+7.5 pp` vs baseline). The unadjusted paired exact McNemar test is `p=0.023703`; because step-2000 was selected from the six-point sweep, the Bonferroni-adjusted value is `0.142216` and should not be treated as confirmatory significance.

### Per-task success counts

| Task | Baseline | FM-500 | FM-1000 | FM-1500 | FM-2000 | FM-2500 | FM-3000 |
|---|---:|---:|---:|---:|---:|---:|---:|
| T1 - pick up the bowl and place it back on the plate | 20/20 | 20/20 | 20/20 | 20/20 | 19/20 | 20/20 | 20/20 |
| T2 - lift the bottle and put it down on the plate | 20/20 | 20/20 | 20/20 | 20/20 | 20/20 | 20/20 | 20/20 |
| T3 - lift the bowl and place it back on the plate 3 times | 20/20 | 20/20 | 20/20 | 20/20 | 19/20 | 20/20 | 20/20 |
| T4 - pick up the bottle and put it down the plate 3 times | 19/20 | 20/20 | 20/20 | 20/20 | 20/20 | 19/20 | 20/20 |
| T5 - lift the bowl and place it back on the plate 5 times | 20/20 | 19/20 | 20/20 | 20/20 | 20/20 | 20/20 | 20/20 |
| T6 - pick up the bowl and place it on the plate 7 times | 6/20 | 11/20 | 13/20 | 12/20 | 16/20 | 13/20 | 15/20 |
| T7 - swap the 2 bowls on their plates using the empty plate | 6/20 | 8/20 | 8/20 | 7/20 | 10/20 | 6/20 | 6/20 |
| T8 - rotate the 3 bowls on their plates from left to right using the empty plate | 11/20 | 8/20 | 10/20 | 9/20 | 14/20 | 10/20 | 10/20 |
| T9 - put the cream cheese in the nearest basket and place that basket in the center | 20/20 | 20/20 | 20/20 | 20/20 | 19/20 | 20/20 | 20/20 |
| T10 - put the cream cheese in the nearest basket and place the empty basket in the center | 20/20 | 20/20 | 20/20 | 20/20 | 20/20 | 20/20 | 20/20 |

The clearest gains at step-2000 are T6 (`6/20` to `16/20`, +50 pp), T7 (`6/20` to `10/20`, +20 pp), and T8 (`11/20` to `14/20`, +15 pp). T1-T3, T5, T9, and T10 have 100% baseline success and therefore no success-rate headroom.

### Per-episode completion steps

The complete 1,400-row episode table contains baseline plus all six FutureMamba checkpoints for every task and episode. `completion_steps` is the recorded task step. Unsuccessful rollouts are retained and normally equal the `max_steps=600` timeout.

- `results/libero_mem_all10_summary.csv`: per-method/checkpoint/task success rates and successful-episode completion-step statistics.
- `results/libero_mem_all10_episode_steps.csv`: every baseline/FutureMamba task, checkpoint, and episode outcome with completion steps.
- `results/libero_mem_all10_completion_steps.md`: human-readable table with all 1,400 episode completion-step sequences.
- `results/libero_mem_all10_final_audit.json`: machine-readable audit export; source audit SHA256 `sha256:c65b7322b0cf38b7a1de9e213d81eb8a2964eb9adb92aafc7ae0062716d90daf`.

Task order follows authoritative `meta/tasks.jsonl`; these are ten independent task models, not one shared model. Checkpoint selection by the same evaluation set is exploratory and optimistic; use the common step-2000 row for the primary comparison.
