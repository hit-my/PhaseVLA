# FutureMamba × RoboMME 纯 PyTorch 实现计划

> **面向 AI 代理的工作者：** 必需子技能：使用 superpowers:subagent-driven-development（推荐）或 superpowers:executing-plans 逐任务实现本计划。步骤使用复选框（`- [ ]`）跟踪。此前 LIBERO 仿真计划已废止。

**目标：** 保留 RoboMME 官方数据、ManiSkill/SAPIEN 仿真器、任务定义、成功判定和评测脚本，只把策略服务替换为冻结 RoboMME `pi05_baseline` + 查询级 Mamba-2 + Uniform-6 Progress Expert 的纯 PyTorch FutureMamba。

**架构：** 策略与仿真使用两个隔离软件环境。策略环境运行当前 OpenPI worktree、PyTorch 基座、Mamba 与 Progress Expert；仿真环境运行固定提交的 `robomme_policy_learning` 及其 `robomme_benchmark` 子模块。RoboMME 官方客户端通过 msgpack/WebSocket 发送 `reset`、`add_buffer`、当前观测并接收 20 步 action chunk；环境默认只执行前 16 步。Mamba 每次 `infer` 只更新一次，`add_buffer` 和底层动作不推进记忆。

**技术栈：** Python 3.11、PyTorch 2.9.1、Triton 3.5.1、官方 Mamba v2.3.2、SafeTensors、RoboMME、ManiSkill/SAPIEN、WebSocket/msgpack、pytest。

**固定来源：**

- `RoboMME/robomme_policy_learning@ecf086c3be7c2223167d9bb2f6ef1f0a6e24353b`
- `RoboMME/robomme_benchmark@856bc3a189d4172f3f47dbee4424d585f8d78db3`
- `state-spaces/mamba@77069de5cdb55cbe98b670889c80df211e031039`
- `Yinpei/robomme_data_h5`
- `Yinpei/pi05_baseline`
- `Yinpei/mme_vla_suite`

**工作区：** `/home/ubuntu/.config/superpowers/worktrees/openpi/futuremamba-pytorch-migration-plan`

**设计规格：** `docs/superpowers/specs/2026-08-10-futuremamba-pytorch-mamba-migration-design.md`

---

## 已完成前置条件

以下已有定向测试证据，后续只在最终验收统一重跑：

- PyTorch 配置与模型工厂；
- 冻结 prefix、Action Expert 单步接口；
- Mamba-2 `MemoryBackend`、reset/snapshot/restore；
- 只读 prefix Progress Expert 原型；
- FutureMamba 硬交接与 episode BPTT 原型；
- conditioning cache、插件 checkpoint 与训练入口；
- PyTorch `FutureMambaPolicy` 状态；
- bundle 严格加载与连接级 WebSocket fork。

这些实现仍包含旧 `progress_depth=4`、past-action memory 输入、旧 handoff/boundary loss 和 LIBERO 配置。本计划负责干净迁移，不把旧合同当作 RoboMME 完成证据。

---

## 目标文件

### 固定外部代码

- 修改：`.gitmodules`
- 新增子模块：`third_party/robomme_policy_learning`
- 递归固定：`third_party/robomme_policy_learning/third_party/robomme_benchmark`

官方 `examples/robomme/eval.py`、`env_runner.py`、`utils.py`、`robomme_benchmark` 任务与成功判定不得复制或重写。

### 当前仓库新增

- `src/openpi/policies/robomme_policy.py`：RoboMME 8 维 joint-angle 输入输出 transforms。
- `src/openpi/policies/robomme_policy_test.py`：payload、shape、dtype 与 20/16 horizon 合同。
- `src/openpi/training/robomme_episode_dataset.py`：按 `epis_idx/step_idx` 构造连续 query 窗口。
- `src/openpi/training/robomme_episode_dataset_test.py`：顺序、burn-in、padding 与跨 episode 拒绝。
- `src/openpi/serving/robomme_protocol_test.py`：官方 reset/add_buffer/infer 协议与连接隔离。
- `scripts/run_robomme_experiment_matrix.py`：分阶段与正式实验命令矩阵。
- `scripts/run_robomme_experiment_matrix_test.py`：任务、seed、split、baseline 与消融完整性。
- `scripts/analyze_robomme_results.py`：四类任务、Overall、95% CI、热力图输入。
- `scripts/analyze_robomme_results_test.py`：聚合与缺失 episode 拒绝。
- `scripts/eval_robomme_memory_swap.py`：同观测、同噪声 memory swap 反事实。
- `scripts/eval_robomme_memory_swap_test.py`：干预变量隔离。

### 当前仓库修改

- `src/openpi/models_pytorch/futuremamba_config.py`
- `src/openpi/models_pytorch/futuremamba_config_test.py`
- `src/openpi/models_pytorch/progress_expert.py`
- `src/openpi/models_pytorch/progress_expert_test.py`
- `src/openpi/models_pytorch/futuremamba.py`
- `src/openpi/models_pytorch/futuremamba_test.py`
- `src/openpi/training/config.py`
- `src/openpi/training/futuremamba_checkpoint.py`
- `src/openpi/training/futuremamba_checkpoint_test.py`
- `src/openpi/policies/futuremamba_policy.py`
- `src/openpi/policies/futuremamba_policy_test.py`
- `src/openpi/policies/policy_config.py`
- `src/openpi/serving/websocket_policy_server.py`
- `src/openpi/serving/websocket_policy_server_test.py`
- `scripts/serve_policy.py`
- `scripts/train_futuremamba_pytorch.py`
- `scripts/train_futuremamba_pytorch_test.py`
- `scripts/validate_jax_pytorch_pi05.py`
- `scripts/profile_futuremamba.py`
- `examples/convert_jax_model_to_pytorch.py`
- `examples/convert_jax_model_to_pytorch_test.py`

### 最终删除

- FutureMamba 专用 JAX 模型、Progress Expert、Mamba 与训练入口；
- `examples/libero_mem/` 中只服务 FutureMamba 的 runner、因果评测和矩阵；
- `memory_backend="mamba"`、`futuremamba_libero_mem` 和 LIBERO 专用 FutureMamba 配置。

OpenPI 上游通用 JAX 支持、普通 `examples/libero/` 和 `third_party/libero` 不属于删除范围。

---

## 阶段 A：冻结 RoboMME 官方闭环

### 任务 1：固定 RoboMME 来源并建立双环境

**文件：** `.gitmodules`、`third_party/robomme_policy_learning`、`environments/futuremamba/pyproject.toml`

- [ ] **步骤 1：添加固定子模块**

```bash
git submodule add https://github.com/RoboMME/robomme_policy_learning.git third_party/robomme_policy_learning
git -C third_party/robomme_policy_learning checkout ecf086c3be7c2223167d9bb2f6ef1f0a6e24353b
git -C third_party/robomme_policy_learning submodule update --init --recursive
```

断言内层 benchmark 为 `856bc3a189d4172f3f47dbee4424d585f8d78db3`；不接受浮动 `main`。

- [ ] **步骤 2：安装 RoboMME 仿真环境**

```bash
micromamba create -n robomme python=3.11 -y
micromamba run -n robomme pip install -r third_party/robomme_policy_learning/examples/robomme/requirements.txt
micromamba run -n robomme pip install -e third_party/robomme_policy_learning/third_party/robomme_benchmark
micromamba run -n robomme pip install -e third_party/robomme_policy_learning/packages/openpi-client
```

- [ ] **步骤 3：运行官方模拟器 smoke**

```bash
micromamba run -n robomme python third_party/robomme_policy_learning/examples/robomme/simple_test.py
```

预期：退出码 0，ManiSkill/SAPIEN 创建环境并完成 step。

- [ ] **步骤 4：记录版本身份**

输出单行 JSON，包含两个 RoboMME commit、Python、ManiSkill、SAPIEN、GPU 与 Vulkan renderer。字段缺失时失败，不写 `unknown` 作为通过值。

### 任务 2：复现官方数据 replay 与 JAX 基线

**文件：** 不修改官方环境代码；输出 `runs/evaluation/pi05_baseline/...`

- [ ] **步骤 1：下载并解压官方数据**

```bash
git clone https://huggingface.co/datasets/Yinpei/robomme_data_h5 data/robomme_data_h5
uv run third_party/robomme_policy_learning/scripts/tarxz_h5.py decompress \
  --input_dir data/robomme_data_h5 --jobs 16 --remove_archive
```

- [ ] **步骤 2：运行官方 replay**

```bash
micromamba run -n robomme python \
  third_party/robomme_policy_learning/third_party/robomme_benchmark/scripts/dataset_replay.py \
  --h5-data-dir data/robomme_data_h5
```

至少保存一个 `PickXtimes` replay 视频与无异常日志。

- [ ] **步骤 3：下载官方微调基座与 assets**

```bash
git clone https://huggingface.co/Yinpei/pi05_baseline runs/ckpts/pi05_baseline
uv run third_party/robomme_policy_learning/scripts/unzip_ckpt.py runs/ckpts/pi05_baseline
```

核对后期 checkpoint 含 `params/`、`assets/robomme/norm_stats.json`。

- [ ] **步骤 4：启动官方 JAX policy server**

```bash
CUDA_VISIBLE_DEVICES=0 uv run third_party/robomme_policy_learning/scripts/serve_policy.py \
  --seed=7 --port=8001 policy:checkpoint \
  --policy.dir=runs/ckpts/pi05_baseline/pi05_baseline/79999 \
  --policy.config=pi05_baseline
```

- [ ] **步骤 5：运行单任务单 episode**

```bash
micromamba run -n robomme python third_party/robomme_policy_learning/examples/robomme/eval.py \
  --args.model_seed=7 --args.port=8001 --args.policy_name=pi05_baseline \
  --args.model_ckpt_id=79999 --args.no-use-history \
  --args.only_tasks=PickXtimes
```

开发 smoke 只运行 episode 0；若官方 CLI 不能限制 episode，新增外层 launcher，不修改 evaluator 的 step、成功判定或动作执行逻辑。

- [ ] **步骤 6：验收产物**

必须存在：视频、动作日志、`progress.json`/`log.json`、任务状态与服务器延迟。没有真实 episode 产物不得进入基座转换。

---

## 阶段 B：RoboMME 基座转换

### 任务 3：注册 RoboMME transforms 与纯 PyTorch 配置

**文件：** 新增 `src/openpi/policies/robomme_policy.py` 及测试；修改 `src/openpi/training/config.py`

- [ ] **步骤 1：写失败测试**

锁定：前视图与腕部图、8 维 state、8 维输出 action、224 × 224 变换、quantile norm、`action_horizon=20`、`execution_horizon=16`、`discrete_state_input=False`。

```python
def test_robomme_output_keeps_joint_angle_action_width():
    out = RoboMMEOutputs()({"actions": np.zeros((20, 32), np.float32)})
    assert out["actions"].shape == (20, 8)
```

- [ ] **步骤 2：实现 `RoboMMEInputs/Outputs`**

语义逐字段对齐固定上游 `robomme_policy.py`；不得借用 LIBERO transforms。

- [ ] **步骤 3：注册配置**

新增 `pi05_robomme_pytorch` 与 `futuremamba_robomme_mamba2`。后者主设置：

```python
FutureMambaPytorchConfig(
    pi05=True,
    discrete_state_input=False,
    action_horizon=20,
    execution_horizon=16,
    progress_depth=6,
    memory_backend="mamba2",
)
```

- [ ] **步骤 4：运行测试**

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=src .venv/bin/pytest -q \
  src/openpi/policies/robomme_policy_test.py \
  src/openpi/models_pytorch/futuremamba_config_test.py
```

### 任务 4：转换 RoboMME `pi05_baseline` 并建立硬门

**文件：** `examples/convert_jax_model_to_pytorch.py`、`scripts/validate_jax_pytorch_pi05.py` 及测试

- [ ] **步骤 1：扩展转换 manifest 测试**

要求 manifest 含 RoboMME policy/benchmark commit、模型 config、RoboMME transforms checksum、`norm_stats.json` checksum、20 步 horizon 和 8 维 action width。

- [ ] **步骤 2：执行 float32 转换**

```bash
PYTHONPATH=src .venv/bin/python examples/convert_jax_model_to_pytorch.py \
  --checkpoint-dir runs/ckpts/pi05_baseline/pi05_baseline/79999 \
  --config-name pi05_robomme_pytorch \
  --output-path runs/ckpts/pi05_baseline_pytorch/79999 \
  --precision float32
```

- [ ] **步骤 3：逐层 parity**

相同 RoboMME observation、action、noise、time 和 seed 比较：prefix last token、Uniform-6 K/V、Action Expert velocity、完整 10-step solver action chunk。

固定门：MAE ≤ `1e-4`，MaxAE ≤ `5e-4`，velocity cosine ≥ `0.9999`。输出 JSON；首个分歧层失败退出。

- [ ] **步骤 4：同批 episode 回归**

对固定的 `PickXtimes` 与 `BinFill` validation episode，分别运行 JAX 与 PyTorch 基座。PyTorch 成功率不得低于 JAX 95% CI 下界；同时记录 action chunk MAE、query 延迟和峰值显存。

---

## 阶段 C：RoboMME 时序训练合同

### 任务 5：实现连续 episode 窗口 Dataset

**文件：** 新增 `src/openpi/training/robomme_episode_dataset.py` 及测试；修改训练配置

- [ ] **步骤 1：写失败测试**

覆盖：按 `(epis_idx, step_idx)` 排序；窗口不跨 episode；step 缺口/重复拒绝；episode 起点零状态；中间窗口需要 burn-in；右侧 padding；20 步 action target。

- [ ] **步骤 2：建立 episode 索引**

从官方 preprocessed pickle 读取 `epis_idx`、`step_idx`、`exec_start_idx`，构建不可变索引。禁止从随机单帧 Dataset 直接训练 Mamba。

- [ ] **步骤 3：实现 truncated BPTT 窗口**

窗口长度 `W` 可配置。中间窗口先从 episode 起点无梯度 burn-in 到窗口起点，再对 W 个 query 建图；窗口末 detach。状态绝不跨 episode。

- [ ] **步骤 4：运行测试**

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=src .venv/bin/pytest -q \
  src/openpi/training/robomme_episode_dataset_test.py
```

### 任务 6：迁移主方法为 VLM-only 查询记忆

**文件：** `futuremamba_config.py`、`futuremamba.py`、Policy 与测试

- [ ] **步骤 1：写失败测试**

```python
def test_memory_input_is_current_vlm_token_only(plugin):
    assert not hasattr(plugin, "executed_action_encoder")
    assert not hasattr(plugin, "memory_input_fusion")
```

另锁定一次 `infer` 恰好一次 memory step；`add_buffer`、16 次底层动作不更新 Mamba；reset 清零。

- [ ] **步骤 2：删除主路径 past-action 输入**

`e_q = W_z z_q`。`pi0.5 + past actions` 作为独立 baseline 实现，不能留在 FutureMamba 默认路径。

- [ ] **步骤 3：迁移 config/checkpoint schema**

以 `prediction_horizon=20`、`execution_horizon=16` 替换旧 `executed_horizon` 语义；metadata 加 RoboMME 三项身份。

- [ ] **步骤 4：运行定向回归**

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=src .venv/bin/pytest -q \
  src/openpi/models_pytorch/futuremamba_config_test.py \
  src/openpi/models_pytorch/futuremamba_test.py \
  src/openpi/policies/futuremamba_policy_test.py \
  src/openpi/training/futuremamba_checkpoint_test.py
```

---

## 阶段 D：Progress Expert 与训练目标

### 任务 7：实现 Uniform-6 Action Expert 初始化

**文件：** `progress_expert.py`、`futuremamba.py` 及测试

- [ ] **步骤 1：写失败测试**

锁定 18 层 Action Expert 的 Uniform-6 映射覆盖首尾；每个对应 block 参数与源层完全相等但不共享 storage；VLM K/V 层索引一致；随机初始化仅显式消融。

- [ ] **步骤 2：实现复制器**

复制 decoder block、AdaRMS、action/time 投影中 shape 对应的权重。Memory projection 等新参数使用标准初始化。复制后重新冻结 base，Progress 参数保持可训练。

- [ ] **步骤 3：迁移默认深度**

`progress_depth=6`；checkpoint metadata 与 bundle loader 严格拒绝旧 4 层 checkpoint 冒充主设置。

- [ ] **步骤 4：运行测试**

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=src .venv/bin/pytest -q \
  src/openpi/models_pytorch/progress_expert_test.py \
  src/openpi/models_pytorch/futuremamba_test.py -k 'initial or mapping or cache'
```

### 任务 8：实现 flow + terminal action loss

**文件：** `futuremamba.py`、训练脚本及测试

- [ ] **步骤 1：写失败测试**

锁定 `$x_K → Action Expert → \hat A$` 的梯度存在；Action Expert 参数无梯度；`terminal_loss_weight=0` 等于 flow-only；非零时 Mamba 与 Progress 参数有梯度；padding 不参与损失。

- [ ] **步骤 2：实现主目标**

```python
flow_loss = masked_mean((progress_velocity - target_velocity).square())
terminal_actions = denoise_from_handoff_with_frozen_parameters(x_k, ...)
terminal_loss = masked_mean((terminal_actions - target_actions).square())
loss = flow_loss + config.terminal_loss_weight * terminal_loss
```

VLM prefix 可 `no_grad()`；terminal Action Expert 只冻结参数，不可 `no_grad()`。

- [ ] **步骤 3：加入显存选项**

支持 Action Expert gradient checkpointing 和预注册的 terminal-loss batch fraction。主实验配置与 metadata 必须记录实际值。

- [ ] **步骤 4：单 batch 过拟合**

固定 `PickXtimes` 小批次，验证 loss 持续下降、base checksum 不变、memory swap 改变早期 velocity。

---

## 阶段 E：RoboMME WebSocket 闭环

### 任务 9：兼容官方 reset/add_buffer/infer 协议

**文件：** WebSocket server、FutureMambaPolicy、RoboMME protocol 测试、`serve_policy.py`

- [ ] **步骤 1：写官方协议失败测试**

发送 `{"reset": true}` 期望 `reset_finished=true`；发送 `add_buffer` 期望 ack 且 query count 不变；发送 infer 期望 20 × 8 actions 且 query count +1。

- [ ] **步骤 2：实现协议路由**

保留当前 `__openpi_control__`，新增官方 RoboMME 路由。每个 connection 使用独立 fork；`add_buffer` 只存诊断 checksum/边界，不推进 Mamba。

- [ ] **步骤 3：扩展服务 metadata**

返回 RoboMME 两个 commit、base checksum、assets checksum、backend/schema、20/16 horizon、Uniform-6 mapping、handoff 与 kernel。

- [ ] **步骤 4：运行测试**

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=src .venv/bin/pytest -q \
  src/openpi/serving/robomme_protocol_test.py \
  src/openpi/serving/websocket_policy_server_test.py \
  packages/openpi-client/src/openpi_client/websocket_client_policy_test.py
```

### 任务 10：完成 Mamba-2 真实 RoboMME smoke

- [ ] **步骤 1：启动 PyTorch server**

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src .venv-futuremamba/bin/python scripts/serve_policy.py \
  --port=8000 policy:checkpoint \
  --policy.config=futuremamba_robomme_mamba2 \
  --policy.dir=runs/ckpts/futuremamba_robomme_mamba2/<step>
```

- [ ] **步骤 2：运行 PickXtimes 与 BinFill**

```bash
micromamba run -n robomme python third_party/robomme_policy_learning/examples/robomme/eval.py \
  --args.port=8000 --args.model_seed=7 --args.policy_name=futuremamba_mamba2 \
  --args.model_ckpt_id=<step> --args.only_tasks=PickXtimes,BinFill
```

开发阶段每任务 10 个 validation episode；需要外层 launcher 选择 split/episode，不修改官方环境与 success 判定。

- [ ] **步骤 3：验收**

每个 episode 必须有视频、action、query index、memory checksum/bytes、handoff step、success/timeout/error。完整 rollout 无 NaN/Inf；reset 后 state 归零；多连接不串线。

---

## 阶段 F：正式实验与论文证据

### 任务 11：建立预注册实验矩阵

**文件：** `run_robomme_experiment_matrix.py` 及测试

矩阵必须包含：

- 主方法：3 个训练 seed；
- baselines：`pi05_baseline`、past actions、MemER、FrameSamp+Modul、FrameSamp+Expert、RMT/TTT 可用变体；
- layer selection：Uniform/First/Sensitivity/Random；
- depth：4/6/9；
- handoff ratio：0.2/0.4/0.6，另有 K=0/K=N；
- memory：full/none/shuffled；
- initialization：Action Expert/random；
- loss：flow-only/flow+terminal；
- Progress Expert：有 memory/无 memory；
- backend：主表 Mamba-2，Mamba-3 只在 gate passed 时单独加入。

测试断言所有 16 个任务、四组分类、50 test episode、seed、checkpoint 和配置 identity 完整且无重复 experiment ID。

### 任务 12：分阶段评测

- [ ] **最小实验：** PickXtimes、BinFill，各 10 validation episode。
- [ ] **Counting Suite：** 4 个任务，各 50 validation episode；预注册主指标为 Counting 平均成功率。
- [ ] **完整验证：** 16 任务 × 50 validation episode。
- [ ] **最终测试：** 16 任务 × 50 test episode × 3 seed；总计每方法 2400 episode。

不得在 test 上调结构或超参数。官方结果可直接引用时记录来源 commit/checkpoint；否则用固定官方 checkpoint 重跑。

### 任务 13：实现结果聚合与可视化数据

**文件：** `analyze_robomme_results.py` 及测试

- 严格要求每个预注册任务/episode 完整；缺失、重复、error 不得按失败静默填充；
- 输出 per-task、Counting/Permanence/Reference/Imitation、Overall；
- 输出 mean、sample std、95% CI 与原始 n；
- 生成 16 任务热力图数据、相对 $\pi_{0.5}$ 的 ΔSR 数据；
- 输出主表所需 trainable/total params、ms/query、chunk latency、峰值显存和 FLOPs。

### 任务 14：实现 memory swap 与机制证据

**文件：** `eval_robomme_memory_swap.py` 及测试

固定当前 observation、physical episode state、prompt、noise 和 solver schedule，只替换两个不同 query 的 `MemorySnapshot`。输出：

- memory progress probe accuracy；
- 每个去噪 step 的 velocity/action error；
- $x_K$ 交接前后误差；
- swap 前后 action chunk distance；
- 分支/停止行为与 episode success。

测试必须证明除 memory checksum 外的所有干预输入 checksum 相同。

### 任务 15：Profile 与视频证据

修改 `profile_futuremamba.py` 使用 bundle loader 和真实 PyTorch tensors，输出：

- total/trainable/plugin params；
- 每层/总 state bytes；
- memory step median/p95；
- 20 步 action chunk median/p95；
- 训练/推理峰值显存；
- episode 平均推理时间；
- 相对冻结 $\pi_{0.5}$ 的 FLOPs；
- GPU/Torch/Triton/CUDA/Mamba/RoboMME commit。

缺失实测项时失败，不写 fake/null 值。同步视频展示 $\pi_{0.5}$、最强官方 memory baseline、FutureMamba，并叠加 query、GT progress、memory progress、当前专家与最终状态。

---

## 阶段 G：Mamba-3 门控与干净切换

### 任务 16：运行 Mamba-3 SISO RTX 5090 十项硬门

沿用已设计的依赖、设备 256-step、10 seeds 前向、10 seeds 反向、sequence/step parity、因果、partial/full reset、固定 state bytes、2→4 checkpoint 和 RoboMME 部署门。输出机器可读 `mamba3_gate.json`。

任一门失败：状态为 `unsupported_on_current_stack`；正式矩阵不生成 Mamba-3 命令；禁止回退 Mamba-2 后标记 Mamba-3。

### 任务 17：删除旧 JAX FutureMamba 与 LIBERO 专用路径

只有 Mamba-2 RoboMME 真实 smoke、checkpoint、训练与 profile 全部通过后执行：

- 删除 FutureMamba 专用 JAX Mamba/Progress Expert/model/train；
- 删除 `examples/libero_mem/` 中 FutureMamba 专用 runner/matrix；
- 删除 `futuremamba_libero_mem` 配置和旧 assets fallback；
- 删除 `memory_backend="mamba"` 与兼容 shim；
- 保留普通 OpenPI JAX 与 LIBERO 示例。

运行 import/search 回归，确保所有生产配置只指向 PyTorch RoboMME FutureMamba。

### 任务 18：最终验收

- [ ] 静态检查与定向测试全部通过；
- [ ] RoboMME replay、JAX baseline、PyTorch parity、单 batch overfit 有证据；
- [ ] Mamba-2 WebSocket 与 PickXtimes/BinFill rollout 通过；
- [ ] Counting、16-task val、16-task test 矩阵完整；
- [ ] baselines、消融、memory swap、profile、视频产物完整；
- [ ] base checksum 始终不变；
- [ ] Mamba-3 按 gate 结果进入或排除；
- [ ] 论文只报告真实运行的后端、任务、seed、episode 与指标。

---

## 停止条件

1. 官方 RoboMME replay/单 episode 不通过：不得开始模型问题排查。
2. JAX → PyTorch 数值或同批 episode parity 不通过：不得训练插件。
3. Dataset 跨 episode、乱序或中间窗口无 burn-in：不得运行正式训练。
4. terminal loss 无法回传到 $x_K$：不得把该配置标为 flow+terminal。
5. PickXtimes/BinFill 真实 smoke 不通过：不得启动 Counting 全量评测。
6. validation 完整前不得触碰 test 超参数选择。
7. Mamba-3 任一硬门失败不阻塞 Mamba-2，但必须从正式矩阵排除。
8. 最终结果缺 episode、seed、checkpoint 或 provenance：不得进入论文表格。
