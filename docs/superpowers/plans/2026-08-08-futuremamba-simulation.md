# FutureMamba 仿真实现计划

> **面向 AI 代理的工作者：** 必需子技能：使用 superpowers:subagent-driven-development（推荐）或 superpowers:executing-plans 逐任务实现此计划。步骤使用复选框（`- [ ]`）语法跟踪进度。

**目标：** 在冻结的任务适配版 $\pi_{0.5}$ 上实现查询级 Mamba 进程记忆与高噪声 Progress Expert 硬交接，并在 LIBERO-Mem / LIBERO-Long 上完成可复现训练、闭环评测和历史因果测试。

**架构：** 复用官方 JAX openpi 的 `Pi0` 作为冻结基座。每个 action-chunk 查询把 VLM 最后一层末端有效 Token 和上一周期实际执行动作输入标准 Mamba 增量状态；冻结 VLM prefix 只计算一次，Progress Expert 与 Action Expert 复用同一份逐层 KV Cache，Memory K/V 只追加到 Progress Expert。Progress Expert 与 Gemma Action Expert 使用同类 block、同宽约 1/4 深度，只接管前 $K=\lceil\rho N\rceil$ 个高噪声 flow step，冻结 Action Expert 从 $x_K$ 接管低噪声精修。训练按完整 episode 展开，仅优化插件参数，损失为高噪声 Flow Matching、交接状态对齐与边界速度对齐。

**技术栈：** openpi 训练/服务使用 Python 3.11、JAX 0.5.3、Flax NNX/Linen、Optax、LeRobot、Orbax、pytest 和 WebSocket/msgpack；LIBERO-Mem 闭环 runner 延续现有 Python 3.8 Docker 环境。

**工作区：** `/home/ubuntu/lgd/futuremamba`，分支 `feature/futuremamba`，基线提交 `650c5b0`。

**已知基线状态：** `uv sync --frozen` 因 PyPI/GitHub TLS 中断未完成；使用另一 openpi 环境与干净环境变量运行离线测试得到 `13 passed`，唯一失败项 `data_loader_test.py::test_with_real_dataset` 需要访问 Hugging Face。实现前必须先完成任务 1，不能把网络失败当作代码失败，也不能在依赖未锁定时修改 `uv.lock`。

**范围：** 本计划只覆盖核心模型、训练、LIBERO-Mem、LIBERO-Long 和仿真因果实验。真机平台与控制仓库尚未确定，按用户决定另立实现计划。

**2026-08-08 实现状态：** 仿真代码与离线验收已完成；FutureMamba 定向回归 180 项通过，全量离线套件拆分验证为 187 项通过、训练 smoke 1 项通过。两个 Hugging Face 联网用例未执行，正式训练主表、真实 LIBERO-Mem rollout 与真机实验仍待运行。

---

## 文件结构

### 新建

- `src/openpi/models/mamba.py`：标准 Mamba-1 selective SSM；提供整段 scan 与单步增量状态。
- `src/openpi/models/mamba_test.py`：scan/step 等价性、固定状态大小、reset 与因果性测试。
- `src/openpi/models/progress_expert.py`：只实例化 Action Expert 一侧参数的同宽减层 Gemma block；直接复用冻结 VLM 的逐层 prefix KV Cache，并只为 Memory Token 生成额外 K/V。
- `src/openpi/models/progress_expert_test.py`：共享缓存、Memory Token 隔离、形状、深度、条件敏感性和「无上下文侧可训练参数」测试。
- `src/openpi/models/futuremamba_config.py`：模型、交接、记忆和损失配置；严格冻结过滤器。
- `src/openpi/models/futuremamba.py`：VLM 特征提取、动作摘要、Mamba 更新、训练损失和双专家采样。
- `src/openpi/models/futuremamba_test.py`：末端 Token、损失、硬交接、$\rho=0/1$ 和状态测试。
- `src/openpi/training/episode_data_loader.py`：按 episode 采样的 LeRobot 数据集、动态 padding 与 query/reset mask。
- `src/openpi/training/episode_data_loader_test.py`：episode 边界、query stride、previous-executed action 和 padding 测试。
- `scripts/train_futuremamba.py`：完整 episode 的插件训练入口。
- `scripts/train_futuremamba_test.py`：两步训练、恢复训练与冻结参数校验。
- `src/openpi/training/weight_loaders_test.py`：部分 checkpoint 合并与最新数值 step 选择测试。
- `src/openpi/policies/futuremamba_policy.py`：外置 Mamba state、executed-action 输入、reset/snapshot/restore。
- `src/openpi/policies/futuremamba_policy_test.py`：连续查询、reset、动作长度和状态快照测试。
- `src/openpi/policies/libero_policy_test.py`：raw action / executed-action 映射一致性测试。
- `examples/libero_mem/convert_to_lerobot.py`：将官方 LIBERO-Mem 和 LIBERO-10/Long 数据转成训练/验证 LeRobot 仓库。
- `examples/libero_mem/convert_to_lerobot_test.py`：内存 episode 的字段、拆分和动作对齐测试。
- `examples/libero_mem/env_adapter.py`：封装官方顺序子目标推进、overshoot reset 和符号谓词事件。
- `examples/libero_mem/env_adapter_test.py`：顺序推进只增量一次、episode reset 和稳定事件测试。
- `examples/libero_mem/metrics.py`：任务成功、子目标完成、重复执行、分支准确率、置信区间和能力保持统计。
- `examples/libero_mem/metrics_test.py`：Sequence/Or、overshoot、重复动作、Wilson 区间和分层聚合测试。
- `examples/libero_mem/main.py`：LIBERO-Mem / LIBERO-Long 闭环 runner 与 JSONL 日志。
- `examples/libero_mem/build_history_pairs.py`：构造同当前输入、不同有效进程的配对清单。
- `examples/libero_mem/eval_history_pairs.py`：状态清零、截断、打乱和交换的因果评测。
- `examples/libero_mem/history_pairs_test.py`：配对筛选、固定输入/噪声和状态干预测试。
- `src/openpi/serving/websocket_policy_server_test.py`：连接级 state 隔离、reset ack 和错误控制消息测试。
- `packages/openpi-client/src/openpi_client/websocket_client_policy_test.py`：客户端 infer/reset 请求串行化测试。
- `examples/libero_mem/run_experiment_matrix.py`：预注册基线、消融、训练 seed 与 rollout 矩阵。
- `examples/libero_mem/run_experiment_matrix_test.py`：矩阵完整性、去重和配置落盘测试。
- `scripts/profile_futuremamba.py`：参数量、state bytes、延迟和显存统计。

### 修改

- `.gitmodules`：将现有 `third_party/libero` 子模块切换到固定提交的 LIBERO-Mem fork；禁止并存两个可导入为 `libero` 的源码树。
- `src/openpi/models/gemma.py`：提取 Action-only 条件 block；原 `Pi0` 数值路径不变。
- `src/openpi/models/pi0.py`：提取可复用的 prefix 编码和冻结 Action Expert 单步速度接口；原 `sample_actions` 保持数值等价。
- `src/openpi/models/pi0_test.py`：增加重构前后同噪声数值一致性测试。
- `src/openpi/training/weight_loaders.py`：增加部分 checkpoint loader 和最新数值 step loader。
- `src/openpi/training/config.py`：注册阶段 A 基座配置和阶段 B 插件配置。
- `src/openpi/training/data_loader.py`：接入阶段 A 的多 suite 平衡 query loader；现有单 repo loader 行为不变。
- `src/openpi/transforms.py`：规范化并 padding `executed_actions`，不污染普通 `actions`。
- `src/openpi/transforms_test.py`：executed-action normalization、padding 与 mask 回归测试。
- `src/openpi/policies/policy_config.py`：FutureMamba checkpoint 使用 `FutureMambaPolicy`。
- `src/openpi/serving/websocket_policy_server.py`：支持 reset 控制消息与连接级 Policy state。
- `src/openpi/policies/policy.py`：`PolicyRecorder` 转发 `fork()` / reset / snapshot / restore 生命周期调用。
- `packages/openpi-client/src/openpi_client/websocket_client_policy.py`：实现真正的远程 `reset()`。
- `packages/openpi-client/src/openpi_client/base_policy.py`：增加 `fork()` / reset 生命周期合同；现有无状态策略行为不变。
- `examples/libero/main.py`：普通 LIBERO 评测也在 episode 开始时 reset，并把实际执行 prefix 传回 stateful policy。
- `examples/libero/Dockerfile`：继续只从 `third_party/libero` 导入环境，并 smoke 检查 `libero_mem` 与 `libero_10` suite。

---

### 任务 1：恢复隔离环境并锁定离线基线

**文件：**
- 不修改源码
- 检查：`pyproject.toml`、`uv.lock`

- [ ] **步骤 1：完成锁文件对应的环境安装**

运行：

```bash
cd /home/ubuntu/lgd/futuremamba
uv sync --frozen
```

预期：退出码 0；`.venv/bin/python` 和 `.venv/bin/pytest` 存在。若仍是 TLS/RPC 中断，只重试下载或配置可用镜像；不要执行 `uv lock`，不要改依赖版本。

- [ ] **步骤 2：在不加载 ROS pytest 插件的环境中运行离线基线**

运行：

```bash
env -i HOME="$HOME" PATH="$PWD/.venv/bin:/usr/bin" \
  PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=src \
  .venv/bin/pytest -q \
  src/openpi/models/pi0_test.py \
  src/openpi/models/model_test.py \
  src/openpi/training/data_loader_test.py \
  -k 'not test_with_real_dataset'
```

预期：全部选中测试通过，0 failed。把 exact pass count 记入实现日志。

- [ ] **步骤 3：确认工作树仍干净**

运行：

```bash
git status --short --branch
```

预期：`feature/futuremamba`，无源码改动。环境文件应已被 `.gitignore` 排除。

---

### 任务 2：实现标准 Mamba 增量状态

**文件：**
- 创建：`src/openpi/models/mamba.py`
- 创建：`src/openpi/models/mamba_test.py`

- [x] **步骤 1：编写失败测试，定义 state/step/scan 合同**

测试至少包含：

```python
def test_step_matches_scan():
    model = SelectiveMamba(MambaConfig(d_model=16, d_state=4, d_conv=3, expand=2, depth=2), nnx.Rngs(0))
    x = jax.random.normal(jax.random.key(1), (2, 7, 16))
    scan_y, scan_state = model.scan(x, model.initial_state(batch_size=2))
    state = model.initial_state(batch_size=2)
    ys = []
    for index in range(x.shape[1]):
        y, state = model.step(x[:, index], state)
        ys.append(y)
    np.testing.assert_allclose(scan_y, jnp.stack(ys, axis=1), rtol=2e-5, atol=2e-5)
    assert jax.tree.all(jax.tree.map(lambda a, b: jnp.allclose(a, b), scan_state, state))


def test_state_size_is_independent_of_sequence_length():
    model = SelectiveMamba(MambaConfig(d_model=16, d_state=4, d_conv=3, expand=2, depth=2), nnx.Rngs(0))
    short = model.scan(jnp.ones((1, 2, 16)), model.initial_state(1))[1]
    long = model.scan(jnp.ones((1, 20, 16)), model.initial_state(1))[1]
    assert jax.tree.structure(short) == jax.tree.structure(long)
    assert jax.tree.map(lambda x: x.shape, short) == jax.tree.map(lambda x: x.shape, long)
```

另测：零状态可重复、改变未来输入不影响过去输出、`d_conv` cache 的时间顺序正确。

- [x] **步骤 2：运行测试确认失败**

运行：

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest -q src/openpi/models/mamba_test.py
```

预期：collection error，`openpi.models.mamba` 不存在。

- [x] **步骤 3：实现 Mamba-1 层与固定大小状态**

公开接口固定为：

```python
@dataclasses.dataclass(frozen=True)
class MambaConfig:
    d_model: int
    d_state: int = 16
    d_conv: int = 4
    expand: int = 2
    depth: int = 2
    dt_rank: int | None = None


@struct.dataclass
class MambaLayerState:
    ssm: jax.Array       # [batch, d_inner, d_state]
    conv: jax.Array      # [batch, d_conv - 1, d_inner]


@struct.dataclass
class MambaState:
    layers: Sequence[MambaLayerState]
```

`SelectiveMamba` 的公开 API 固定为：`initial_state(batch_size: int, dtype=jnp.float32) -> MambaState`、`step(x: jax.Array, state: MambaState) -> tuple[jax.Array, MambaState]`、`scan(x: jax.Array, state: MambaState) -> tuple[jax.Array, MambaState]`。`step` 接收 `[B,D]`，`scan` 接收 `[B,T,D]`；二者返回的最后状态必须同构。

每层严格实现选择性 SSM：`A=-exp(A_log)`、输入依赖的 `dt/B/C`、`dA=exp(dt*A)`、`state=dA*state+dt*B*x`、`y=C·state+D*x`，并在 SSM 前使用 depthwise causal convolution、SiLU gate、residual 和 RMSNorm。`scan` 必须通过同一个 `step` 使用 `jax.lax.scan`，保证训练与在线推理定义一致。

- [x] **步骤 4：运行 Mamba 测试**

运行同步骤 2。预期：全部通过。

- [x] **步骤 5：提交**

```bash
git add src/openpi/models/mamba.py src/openpi/models/mamba_test.py
git commit -m "feat: add streaming selective Mamba memory"
```

---

### 任务 3：实现复用冻结前缀缓存的轻量 Progress Expert

**文件：**
- 创建：`src/openpi/models/progress_expert.py`
- 创建：`src/openpi/models/progress_expert_test.py`
- 修改：`src/openpi/models/gemma.py`

- [ ] **步骤 1：测试共享缓存与动作分支隔离合同**

固定调用合同：

```text
ProgressExpert(
    prefix_kv_cache: KVCache,             # [L_A,B,C,K,H]，来自冻结 VLM 的同一次 prefix forward
    prefix_mask: bool[B,C],
    memory_token: float[B,1,D_M],
    noisy_actions: float[B,H,A],
    timestep: float[B],
    use_prefix_cache: bool = True,
) -> velocity: float[B,H,A]
```

核心测试覆盖：输出与 noisy action 同形；`progress_depth` 生效；改变 Memory Token 会改变速度；改变未 mask 的共享 prefix KV 会改变速度；改变被 mask 的 prefix KV 不影响输出；`use_prefix_cache=False` 时改变 prefix KV 不影响输出；Memory Token 不改变输入 cache；参数树中不存在 context query、context output projection、context FFN 或第二套 VLM context K/V projection。

使用可识别的逐层 fake cache 验证层映射 $r(j)$：默认在 Action Expert 的 $L_A$ 层上等间隔选取 $L_P$ 个索引，首尾层均被覆盖，索引写入 config/checkpoint metadata。另测 Memory K/V 追加后，action query 的位置在有效 prefix 与 Memory Token 之后；prefix padding 仍被 mask。

- [ ] **步骤 2：运行测试确认失败**

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest -q src/openpi/models/progress_expert_test.py
```

- [ ] **步骤 3：实现 Action-only 轻量 Gemma Expert**

Progress Expert 不能创建 `_gemma.Module(configs=[context_config, action_config])`，也不能从 `prefix_out` 重新训练 context K/V projection。应在 `gemma.py` 提取 `CachedPrefixActionBlock`：只包含 Action Expert 一侧的 AdaRMSNorm、action Q/K/V、attention output、FFN、residual，以及 Memory Token 的 K/V projection。

每个 Progress block 的 key/value 为 `[K_prefix_cache[r(j)]; K_memory[j]; K_action[j]]` / `[V_prefix_cache[r(j)]; V_memory[j]; V_action[j]]`，query 只来自 action tokens。`prefix_mask`、单个有效 Memory 位置和 action block-causal mask 共同构成 attention mask。共享 prefix cache 全程 `stop_gradient`；Memory K/V 与 action-side 参数可训练。Progress Expert 深度用 `progress_depth`，Action hidden width、head 数、head dim、MLP dim和 AdaRMS 时间条件均与原 Action Expert 相同。

`gemma.py` 只做无行为变化的组件提取；原 `_gemma.Module` 继续使用原 `Block` 路径。回归测试必须证明现有 Pi0 输出不变。

- [ ] **步骤 4：运行 Progress Expert 与 Gemma 回归测试**

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest -q \
  src/openpi/models/progress_expert_test.py \
  src/openpi/models/model_test.py
```

- [ ] **步骤 5：提交**

```bash
git add src/openpi/models/gemma.py src/openpi/models/progress_expert.py src/openpi/models/progress_expert_test.py
git commit -m "feat: add cache-conditioned progress expert"
```

---

### 任务 4：定义 FutureMamba 配置、冻结边界和 checkpoint 初始化

**文件：**
- 创建：`src/openpi/models/futuremamba_config.py`
- 创建：`src/openpi/models/futuremamba.py`
- 创建：`src/openpi/models/futuremamba_test.py`
- 创建：`src/openpi/training/weight_loaders_test.py`
- 修改：`src/openpi/training/weight_loaders.py`
- 修改：`src/openpi/models/pi0.py`
- 修改：`src/openpi/models/pi0_test.py`

- [ ] **步骤 1：为配置和冻结过滤器写失败测试**

```python
def test_only_futuremamba_parameters_are_trainable():
    config = FutureMambaConfig(
        paligemma_variant="dummy",
        action_expert_variant="dummy",
        progress_depth=1,
        memory=MambaConfig(d_model=64, d_state=4, d_conv=3, depth=1),
    )
    model = nnx.eval_shape(config.create, jax.random.key(0))
    trainable = nnx.state(model, config.get_trainable_filter()).flat_state()
    assert trainable
    assert all(path[0] == "futuremamba" for path in trainable)
```

同时测试：`pi05` 和 `discrete_state_input` 必须为真、`0 <= handoff_ratio <= 1`、`1 <= progress_depth <= action_expert_depth`、`num_denoise_steps > 0`、`executed_horizon <= action_horizon`、`handoff_loss_weight >= 0`、`boundary_loss_weight >= 0`；`progress_prefix_layer_indices` 必须严格递增、长度等于 `progress_depth` 且均落在 Action Expert 层范围；`progress_depth` 的主设置由 `round(action_expert_depth / 4)` 得到（`gemma_300m` 为 4），消融比例也必须在构造配置前转换为至少 1 的整数层数。

- [ ] **步骤 2：测试并实现严格的部分 checkpoint 加载器**

构造一个只有 `Pi0` 参数的临时 checkpoint，调用新的 `PartialCheckpointWeightLoader(params_path, missing_regex=r"futuremamba/.*")`。该类在 `weight_loaders.py` 中复用 `_merge_params`，但必须先比较扁平 key、shape 和 dtype：只允许 `futuremamba/.*` 缺失并保留当前初始化；任何其他缺失键、多余键或 shape 不匹配立即报错。测试断言基座参数来自 checkpoint、插件参数保持初始化，并覆盖 3 类拒绝路径。

- [ ] **步骤 3：重构 Pi0 prefix/velocity helper 并证明数值不变**

在 `Pi0` 中提取两个具体方法：`encode_prefix(observation)` 返回 `prefix_out`、`prefix_mask`、`prefix_ar_mask` 和完整逐层 `_gemma.KVCache`；`action_velocity(observation, x_t, timestep, prefix_mask, kv_cache)` 返回 `[B,H,A]` 速度场。`prefix_out` 只供 Mamba 选择/压缩记忆输入；同一份 `kv_cache` 由冻结 Action Expert 原样读取，并由 Progress Expert 按 `progress_prefix_layer_indices` 选层读取。方法名和返回顺序必须由 `Pi0` 与 `FutureMamba` 共用，禁止复制去噪或再次运行 prefix。

原 `sample_actions` 改为调用 helper。使用固定 fake observation 与固定 noise，断言重构前保存的 golden 输出和重构后输出在 `rtol=1e-6, atol=1e-6` 下相同；增加计数器证明一次 action query 只执行一次 prefix forward。

- [ ] **步骤 4：创建 FutureMambaConfig 与模型骨架**

`FutureMambaConfig` 继承 `Pi0Config`，强制 `pi05=True` 与 `discrete_state_input=True`；本体状态经 Pi05 tokenizer 进入语言条件，不能因继承现有无状态 LIBERO 示例而丢弃。新增：

```python
progress_depth: int = 4
progress_prefix_layer_indices: tuple[int, ...] | None = None  # None 时等间隔映射并覆盖首尾层
handoff_ratio: float = 0.2
num_denoise_steps: int = 10
executed_horizon: int = 5
executed_action_noise_std: float = 0.01  # 归一化 action 单位，仅训练期作用于记忆输入
handoff_loss_weight: float = 1.0
boundary_loss_weight: float = 0.1
memory: MambaConfig = MambaConfig(d_model=1024)
memory_input: Literal["token_action", "token_only"] = "token_action"
conditioning_pool: Literal["last_valid", "attention", "tokens4", "tokens8"] = "last_valid"
memory_backend: Literal["mamba", "gru", "lstm", "frame_stack", "none"] = "mamba"
frame_stack_window: int = 4
bptt_window_queries: int | None = None
decoder_mode: Literal["handoff", "action_memory_full"] = "handoff"
coupling: Literal["hard", "convex", "residual"] = "hard"
use_prefix_cache: bool = True
reset_memory_every_query: bool = False
```

`FutureMamba` 继承 `Pi0`，保证基座参数路径仍是 `PaliGemma/<subtree>`、`action_in_proj/<subtree>` 等；所有新增组件封装在唯一属性 `self.futuremamba` 下。`get_trainable_filter()` 返回 `nnx.All(nnx.Param, PathRegex(r"futuremamba/.*"))`，`get_freeze_filter()` 返回 `nnx.All(nnx.Param, nnx.Not(PathRegex(r"futuremamba/.*")))`。任务 7 注册 `TrainConfig` 时必须显式令 `freeze_filter=model.get_freeze_filter()`；测试同时从 config 与最终 `TrainConfig.trainable_filter` 枚举路径，禁止只测试模型 helper。

- [ ] **步骤 5：运行配置、loader 与 Pi0 回归测试**

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest -q \
  src/openpi/models/pi0_test.py \
  src/openpi/models/futuremamba_test.py \
  src/openpi/training/weight_loaders_test.py
```

- [ ] **步骤 6：提交**

```bash
git add src/openpi/models/pi0.py src/openpi/models/pi0_test.py \
  src/openpi/models/futuremamba.py src/openpi/models/futuremamba_config.py \
  src/openpi/models/futuremamba_test.py src/openpi/training/weight_loaders.py \
  src/openpi/training/weight_loaders_test.py
git commit -m "feat: define frozen FutureMamba model boundary"
```

---

### 任务 5：实现完整 Episode 数据加载

**文件：**
- 创建：`src/openpi/training/episode_data_loader.py`
- 创建：`src/openpi/training/episode_data_loader_test.py`
- 修改：`src/openpi/training/config.py`

- [ ] **步骤 1：定义 EpisodeBatch 行为测试**

使用内存 fake dataset，两个 episode 分别 12 帧和 7 帧；`query_stride=5`、`action_horizon=10`。断言：

```python
assert batch.actions.shape == (2, 3, 10, action_dim)
assert batch.executed_actions.shape == (2, 3, 5, action_dim)
np.testing.assert_array_equal(batch.query_mask[0], [True, True, True])
np.testing.assert_array_equal(batch.query_mask[1], [True, True, False])
np.testing.assert_array_equal(batch.reset_mask[0], [True, False, False])
np.testing.assert_array_equal(batch.reset_mask[1], [True, False, False])
assert not batch.executed_action_mask[:, 0].any()
np.testing.assert_allclose(batch.executed_actions[0, 1], batch.actions[0, 0, :5])
```

另测 action horizon 的末尾 padding 不跨 episode、不同 episode 不共享 previous actions、collate 只在 query 维 padding。`BalancedQueryDataset` 必须按配置的 suite 权重先选 LIBERO-Mem / LIBERO-Long，再均匀选 task、episode 和该 episode 内 query；固定 seed 的索引序列可复现。

- [ ] **步骤 2：运行测试确认失败**

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest -q src/openpi/training/episode_data_loader_test.py
```

- [ ] **步骤 3：实现 episode-first 数据合同**

```python
@struct.dataclass
class EpisodeBatch:
    observation: _model.Observation   # 每个叶子 [B, Q, *observation_dims]
    actions: jax.Array                # [B, Q, H, D]
    action_mask: jax.Array            # [B, Q, H]
    executed_actions: jax.Array       # [B, Q, E, D]
    executed_action_mask: jax.Array   # [B, Q, E]
    query_mask: jax.Array             # [B, Q]
    reset_mask: jax.Array             # [B, Q]，每个 episode 仅首个有效 query 为真
    episode_index: jax.Array          # [B]
```

`LeRobotEpisodeDataset` 读取 `episode_data_index["from"/"to"]`，以 `query_stride` 构造 query index；底层 `delta_timestamps` 只为目标 `actions` 建立未来 horizon。先对每个 query 应用现有 LIBERO repack、normalize、tokenize、pad transforms，再令第 $q>0$ 个 query 的 executed prefix 等于前一 query 目标动作前 `query_stride` 个有效动作；第一个 query 全零且 mask 为假。

`EpisodeCollator` 只 padding query 维，并产生 `query_mask` 与 `reset_mask`。插件训练 sampler 以 episode 为单位；阶段 A 使用 `BalancedQueryDataset` 做单查询训练。两者都先按配置 suite 权重采样，再在 suite 内均匀采样 task 和 episode，禁止长 episode 仅因帧数更多获得额外权重。

- [ ] **步骤 4：运行 loader 测试**

运行同步骤 2。预期：全部通过，无网络访问。

- [ ] **步骤 5：提交**

```bash
git add src/openpi/training/episode_data_loader.py \
  src/openpi/training/episode_data_loader_test.py src/openpi/training/config.py
git commit -m "feat: load ordered action-chunk episodes"
```

---

### 任务 6：实现记忆更新、训练损失和硬交接采样

**文件：**
- 修改：`src/openpi/models/futuremamba.py`
- 修改：`src/openpi/models/futuremamba_test.py`

- [ ] **步骤 1：测试记忆汇聚与 padding 无关**

对同一个 prompt 分别 padding 到 32 和 64；调用 `_encode_memory_inputs(prefix_out, prefix_mask, mode)`，断言 `last_valid`、`attention`、`tokens4` 和 `tokens8` 的有效输出不受 padding 影响。`last_valid` 必须找 `prefix_mask` 中最右侧真值，不能用 `sum(mask)-1`；多 Token 模式使用 learned queries 对有效 prefix 做压缩，不读取官方 object mask。

- [ ] **步骤 2：测试时间方向、共享缓存、mask、动作扰动和 episode reset**

断言 sampled training time 满足 `t >= 1-K/N`；目标为 `noise-actions`；padding query 的 FM/handoff/boundary loss 均为 0；同一 query 的 Progress Expert 与 Action Expert 收到对象 identity 相同的冻结 prefix cache；Memory K/V 只追加到 Progress Expert；`reset_mask=True` 在该 query 编码前清零 state；两个 episode 的第二个样本输出不受第一个样本历史影响。固定 RNG 时，仅 `add_executed_action_noise=True` 对 executed-action 摘要加入 `executed_action_noise_std` 高斯扰动；冻结 observation 预处理始终 `train=False`。

- [ ] **步骤 3：测试硬切换合同**

使用可计数 fake velocity functions：

```python
result, trace = integrate_handoff(
    noise=jnp.zeros((1, 10, 32)),
    num_steps=10,
    handoff_steps=2,
    progress_velocity=lambda x, t: jnp.ones_like(x),
    action_velocity=lambda x, t: 2 * jnp.ones_like(x),
)
assert trace.progress_calls == 2
assert trace.action_calls == 8
```

增加：`rho=0` 与父类 `Pi0.sample_actions` 在相同 observation/noise 下数值一致；`rho=1` 不调用 Action Expert；$x_K$ 不重采样。

- [ ] **步骤 4：实现按 query 扫描的单次冻结 VLM 编码与动作摘要**

沿 query 维使用 `jax.lax.scan`；冻结 prefix/Action Expert 路径无条件调用 `preprocess_observation(None, observation_q, train=False)`。每个 query 只调用一次 `encode_prefix`，立即对 `prefix_out`、`prefix_mask` 和完整逐层 KV Cache 使用 `jax.lax.stop_gradient`。禁止把整条 episode 展平为 `[B*Q,*observation_dims]`。动作摘要使用共享两层 MLP、masked mean、最后有效动作编码和线性融合；仅 `add_executed_action_noise=True` 且 `memory_input="token_action"` 时加入配置化噪声。Progress Expert 直接读取选层后的共享冻结 prefix cache，并只为 Memory Token 生成额外 K/V；不能重跑 VLM 或训练另一套 context K/V。

- [ ] **步骤 5：实现 Mamba 完整展开与 Memory Token**

以 query 为 scan 轴；先按 batch item 应用 reset，再执行 memory step，最后冻结无效 query：

```python
def _select_batch(mask: jax.Array, new: jax.Array, old: jax.Array) -> jax.Array:
    shaped = mask.reshape((mask.shape[0],) + (1,) * (new.ndim - 1))
    return jnp.where(shaped, new, old)

zero_state = self.futuremamba.memory.initial_state(batch_size)
state_in = jax.tree.map(lambda zero, old: _select_batch(reset_mask_q, zero, old), zero_state, state)
h_q, proposed_state = self.futuremamba.memory.step(memory_input_q, state_in)
state = jax.tree.map(lambda new, old: _select_batch(query_mask_q, new, old), proposed_state, state_in)
memory_token_q = self.futuremamba.memory_token_proj(h_q)
```

每个 batch item 从零状态开始。无效 query 不更新 state，也不进入 loss。

- [ ] **步骤 6：实现 FM、handoff 与 boundary loss**

公开 `compute_episode_loss(rng: at.KeyArrayLike, batch: EpisodeBatch, *, add_executed_action_noise: bool = False) -> dict[str, jax.Array]`。返回键固定为 `loss`、`flow_loss`、`handoff_loss`、`handoff_error`、`boundary_loss` 和 `boundary_error`。高噪声时间从原 Beta(1.5,1) 截断分布采样；handoff 从同一噪声按实际 Euler solver 滚动 $K$ 步；boundary 在 `stop_gradient(x_K)` 上比较 Progress velocity 与冻结 Action velocity，后者也 `stop_gradient`。每个 query 先按 `action_mask` 取均值，再以 `query_mask` 对每条 episode 取均值，最后对 batch 内 episode 等权取均值。分别测试 `handoff_loss_weight=0` 与 `boundary_loss_weight=0`；若 boundary 阻止不同历史产生必要的不同早期速度，允许主配置将其置零。

- [ ] **步骤 7：实现外置 state 的采样接口**

模型公开以下两个确定接口：

```text
initial_memory_state(batch_size: int) -> MambaState

sample_actions_with_memory(
    rng,
    observation,
    memory_state,
    executed_actions,
    executed_action_mask,
    num_steps: int | None = None,
    handoff_ratio: float | None = None,
    noise=None,
) -> (actions: float[B,H,A], next_state: MambaState, diagnostics: dict[str, Array])
```

`sample_actions()` 作为兼容入口，使用零 state 和空 executed-action；stateful 部署只调用 `sample_actions_with_memory()`。

- [ ] **步骤 8：运行模型测试**

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest -q \
  src/openpi/models/mamba_test.py \
  src/openpi/models/progress_expert_test.py \
  src/openpi/models/futuremamba_test.py \
  src/openpi/models/pi0_test.py
```

- [ ] **步骤 9：提交**

```bash
git add src/openpi/models/futuremamba.py src/openpi/models/futuremamba_test.py
git commit -m "feat: train and sample noise-time expert handoff"
```

---

### 任务 7：接入两阶段训练与恢复

**文件：**
- 创建：`scripts/train_futuremamba.py`
- 创建：`scripts/train_futuremamba_test.py`
- 修改：`src/openpi/training/config.py`
- 修改：`src/openpi/training/data_loader.py`
- 修改：`src/openpi/training/weight_loaders.py`
- 测试：`src/openpi/training/weight_loaders_test.py`
- [ ] **步骤 1：注册确定性的两阶段配置与归一化资产**

- `pi05_futuremamba_base`：标准 `Pi0Config(pi05=True, action_horizon=10, discrete_state_input=True)`；读取 `futuremamba/libero_mem_long_train`；使用任务 5 的 `BalancedQueryDataset`，suite 权重固定为 LIBERO-Mem 0.5 / LIBERO-Long 0.5；初始化 `gs://openpi-assets/checkpoints/pi05_base/params`；训练输出约定为 `checkpoints/pi05_futuremamba_base/base/`。
- `futuremamba_libero_mem`：`FutureMambaConfig(action_horizon=10, executed_horizon=5, num_denoise_steps=10, handoff_ratio=0.2, progress_depth=4, discrete_state_input=True)`；任务 7 实现 `LatestCheckpointWeightLoader(checkpoint_root)`：枚举 root 下 Orbax 数值 step，忽略非数值目录，选择最大 step 的 `<step>/params`，空 root 明确报错，再委托任务 4 的严格 partial loader，只允许插件参数缺失。读取同一 train repo，但由 `LeRobotEpisodeDataset` 按相同 suite 权重采完整 episode；`freeze_filter` 必须显式设为该 model config 的 `get_freeze_filter()`。

转换完成后先运行 `uv run scripts/compute_norm_stats.py --config-name pi05_futuremamba_base`，将合并 train repo 的 `state/actions` 统计写入该 repo 对应 asset ID。阶段 A checkpoint 必须复制该资产；阶段 B 的 data config 复用完全相同的 asset ID 和统计文件，不能针对 val/test 重算。测试最大 step 选择、空目录报错、非数值目录忽略、非插件缺失/多余/shape 不匹配报错、两个阶段 norm stats checksum 相同、最终 `TrainConfig.trainable_filter` 只命中 `futuremamba/.*`，以及固定 seed 下两个 suite 均不会因 episode 长度获得额外采样权重。

- [ ] **步骤 2：编写两步插件训练失败测试**

使用 dummy PaliGemma/Action Expert、fake `EpisodeBatch`、2 个 train step。保存训练前基座参数 checksum；训练后断言：插件参数改变，基座 checksum 不变，checkpoint 可恢复并继续到第 4 步。

- [ ] **步骤 3：实现 episode trainer**

复用 `scripts/train.py` 的 mesh、FSDP、optimizer、EMA 和 Orbax 模式，但 batch 类型改为 `EpisodeBatch`，loss 调用 `compute_episode_loss(train_rng, batch, add_executed_action_noise=True)`。日志固定包含：

```python
{
    "loss": loss,
    "flow_loss": losses["flow_loss"],
    "handoff_loss": losses["handoff_loss"],
    "handoff_error": losses["handoff_error"],
    "grad_norm": optax.global_norm(grads),
    "trainable_param_count": trainable_param_count,
}
```

训练启动时枚举 `config.trainable_filter` 命中的所有 parameter path；发现任何不以 `futuremamba/` 开头的路径立即报错。模型整体调用 `eval()`，冻结 prefix/Action Expert 路径始终以 `train=False` 预处理；只有 `add_executed_action_noise=True` 控制插件动作摘要扰动。所有冻结输出在进入插件前 `stop_gradient`，优化器 state 只由 trainable subtree 初始化。

- [ ] **步骤 4：运行训练测试**

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 JAX_PLATFORMS=cpu uv run pytest -q scripts/train_futuremamba_test.py
```

- [ ] **步骤 5：提交**

```bash
git add scripts/train_futuremamba.py scripts/train_futuremamba_test.py \
  src/openpi/training/config.py src/openpi/training/data_loader.py \
  src/openpi/training/weight_loaders.py src/openpi/training/weight_loaders_test.py
git commit -m "feat: train FutureMamba on complete episodes"
```

---

### 任务 8：实现 stateful Policy 与远程 reset

**文件：**
- 创建：`src/openpi/policies/futuremamba_policy.py`
- 创建：`src/openpi/policies/futuremamba_policy_test.py`
- 创建：`src/openpi/policies/libero_policy_test.py`
- 创建：`src/openpi/serving/websocket_policy_server_test.py`
- 创建：`packages/openpi-client/src/openpi_client/websocket_client_policy_test.py`
- 修改：`src/openpi/transforms.py`
- 修改：`src/openpi/transforms_test.py`
- 修改：`src/openpi/policies/libero_policy.py`
- 修改：`src/openpi/policies/policy.py`
- 修改：`src/openpi/policies/policy_config.py`
- 修改：`src/openpi/serving/websocket_policy_server.py`
- 修改：`packages/openpi-client/src/openpi_client/base_policy.py`
- 修改：`packages/openpi-client/src/openpi_client/websocket_client_policy.py`
- 修改：`examples/libero/main.py`

- [ ] **步骤 1：测试 executed-action 变换**
输入 raw LIBERO 7D `executed_actions`。扩展 `LiberoInputs`，只把该字段按 raw `actions` 的相同 data-space 语义复制为 7D；在 `policy_config.create_trained_policy()` 的输入链中，顺序固定为 `LiberoInputs` → 通用 `Normalize` → `NormalizeExecutedActions(norm_stats["actions"])` → `PadExecutedActions(executed_horizon, action_dim)` → model transforms。最后两项分别复用 `actions` 的 quantile/z-score 统计和产生 `[executed_horizon, action_dim]` + mask。测试同一个 raw action 分别走 target-action 与 executed-action 路径时前 7 维逐元素相同、尾部为零；普通 `actions` target 和输出反归一化行为不变。

- [ ] **步骤 2：测试 Policy 生命周期**

连续两次 `infer` 后 state 改变；`reset()` 后 state 与 `model.initial_memory_state(1)` 完全一致；超过 `executed_horizon` 报错；不足长度只在尾部 padding；`snapshot_state()` / `restore_state()` round-trip 后同 observation/noise 产生同动作。

- [ ] **步骤 3：实现 FutureMambaPolicy**

`infer` 先复制 raw observation，再让完整输入链同时变换 observation 与 `executed_actions`；变换完成后弹出 `executed_actions` / `executed_action_mask`，其余字段传给 `Observation.from_dict()`。禁止在变换前弹出，也禁止把新 key 直接交给通用 `Normalize`：norm stats 没有 `executed_actions` 同名键。模型 state 始终存放在 Policy，不写入 NNX Module，避免 `module_jit` 丢弃 mutation。

公开接口固定为：

```text
FutureMambaPolicy.infer(obs: dict, noise: ndarray | None = None) -> dict
FutureMambaPolicy.reset() -> None
FutureMambaPolicy.snapshot_state() -> MambaState
FutureMambaPolicy.restore_state(state: MambaState) -> None
FutureMambaPolicy.fork() -> FutureMambaPolicy
```

`fork()` 共享冻结模型、JIT 函数、transforms 和 metadata，但创建独立零 memory state 与独立 RNG。根 Policy 通过单调 session counter 对初始 key 做 `jax.random.fold_in`，禁止多个连接共享或竞争同一可变 RNG。输出除 `actions` 外包含 `handoff_step`、`memory_state_bytes` 和 timing；不通过网络返回完整 state。

- [ ] **步骤 4：实现 WebSocket reset 控制消息**

协议固定为客户端发送：

```python
{"__openpi_control__": "reset"}
```

服务端调用 `policy.reset()` 并返回：

```python
{"reset": True}
```

`BasePolicy.fork()` 默认返回自身，保持现有无状态策略兼容；`FutureMambaPolicy.fork()` 返回独立 state。`PolicyRecorder` 必须在 `fork()` 时包装底层会话 Policy，并把 reset/snapshot/restore 转发给底层，避免 `serve_policy.py --record` 退化为共享状态。服务端 `_handler` 在发送 metadata 前创建 `session_policy = self._policy.fork()`。该连接的 infer/reset 只访问 `session_policy`，禁止用新连接 reset 其他客户端。协议测试并发建立 2 个客户端，证明 A 的 infer/reset 不改变 B state；未知 control 值返回结构化错误。`WebsocketClientPolicy.reset()` 必须复用同一连接，发送消息、等待 ack，并用与 `infer` 共用的 mutex 保证同步 client 不交错收包。

- [ ] **步骤 5：让 LIBERO rollout 回传实际执行动作**

`examples/libero/main.py` 每个 episode 在 `env.reset()` 后调用 `client.reset()`；每次新 query 把上一周期实际下发的 action prefix 放入 `element["executed_actions"]`。第一次 query 传形状 `(0,7)` 的空数组。

- [ ] **步骤 6：运行策略和协议测试**

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest -q \
  src/openpi/policies/futuremamba_policy_test.py \
  src/openpi/policies/libero_policy_test.py src/openpi/transforms_test.py \
  src/openpi/policies/policy_test.py \
  src/openpi/serving/websocket_policy_server_test.py \
  packages/openpi-client/src/openpi_client/websocket_client_policy_test.py
```

- [ ] **步骤 7：提交**

```bash
git add src/openpi/policies/futuremamba_policy.py \
  src/openpi/policies/futuremamba_policy_test.py src/openpi/policies/libero_policy.py \
  src/openpi/policies/libero_policy_test.py src/openpi/transforms.py src/openpi/transforms_test.py \
  src/openpi/policies/policy.py src/openpi/policies/policy_config.py \
  src/openpi/serving/websocket_policy_server.py src/openpi/serving/websocket_policy_server_test.py \
  packages/openpi-client/src/openpi_client/base_policy.py \
  packages/openpi-client/src/openpi_client/websocket_client_policy.py \
  packages/openpi-client/src/openpi_client/websocket_client_policy_test.py \
  examples/libero/main.py
git commit -m "feat: preserve progress memory across policy queries"
```

---

### 任务 9：接入 LIBERO-Mem 数据与官方顺序环境

**文件：**
- 修改：`.gitmodules`
- 修改：`examples/libero/Dockerfile`
- 修改：`examples/libero/requirements.in`
- 修改：`examples/libero/requirements.txt`
- 创建：`examples/libero_mem/convert_to_lerobot.py`
- 创建：`examples/libero_mem/convert_to_lerobot_test.py`
- 创建：`examples/libero_mem/env_adapter.py`
- 创建：`examples/libero_mem/env_adapter_test.py`

- [ ] **步骤 1：初始化并以单一 import root 固定 LIBERO-Mem fork**

```bash
git submodule set-url third_party/libero https://github.com/libero-mem/libero-mem.git
git submodule sync -- third_party/libero
git submodule update --init --recursive third_party/libero
git -C third_party/libero fetch origin 0eee0defa4024e51511b6e15a79f30e9620209de
git -C third_party/libero checkout 0eee0defa4024e51511b6e15a79f30e9620209de
```

禁止另建 `third_party/libero_mem`：两个目录都导出顶层包 `libero`，会让训练与 runner 随 `PYTHONPATH` 顺序加载不同实现。固定 fork 没有根目录 `requirements.txt`：从 Dockerfile 删除对 `/tmp/requirements-libero.txt` 的 COPY/sync；在 `examples/libero/requirements.in` 显式加入 `bddl==1.0.1`、`easydict==1.9`、`cloudpickle==2.1.0`、`gym==0.25.2`，用文件头记录的 `uv pip compile` 命令重生成 `requirements.txt`。容器仍只从 `/app/third_party/libero` 导入，构建后 smoke 检查 `libero_mem` 和 `libero_10` suite 均可发现。

- [ ] **步骤 2：封装并测试官方顺序进度接口**

`LiberoMemEnvAdapter.reset()` 在 `env.reset()` 后调用 `reset_subgoal_progress()`，并显式将官方遗漏重置的 `_overshot` 清为 `False`。每个物理 step 后恰好调用一次 `_check_success(inc=True)`；`env.step()` 内部的无增量检查不能替代或重复这次调用。adapter 返回 `success`、`get_satisfied_subgoals(task_text)`、`overshot` 和各原子谓词真值。fake env 测试必须证明计数器每 step 只增加 1、reset 后 overshoot 不泄漏到下一 episode。

- [ ] **步骤 3：测试官方数据 → LeRobot 映射和任务提示合同**

内存构造一个 8D state、7D action、主视角/腕部图像、`is_first/is_last` 和语言指令的 episode。写入时调用 `dataset.add_frame(frame, task=instruction)`；保存并重新加载后，必须验证每帧含 `task_index`，且该索引在 `dataset.meta.tasks` 中精确映射回原指令，使 `PromptFromLeRobotTask` 可直接运行。另断言 episode 边界保留、图像不重复旋转、动作长度不变；`image_reasoning` 只进入离线元数据，绝不进入模型 prompt。

- [ ] **步骤 4：实现可复现的本地训练/验证协议**

官方固定提交的 RLDS builder 只声明 `train` split，并在 demo ID 上使用未排序的 set；因此不能声称存在官方 validation split。转换器先按数值 demo ID 排序，再执行预注册的本地拆分：LIBERO-Mem 每 task 前 100 条 train、后 20 条 val；LIBERO-10 每 task 前 40 条 train、后 10 条 val。若下载数据的每 task 数量不满足该合同，转换器立即报错并打印实际数量，不静默改比例。输出：

```text
futuremamba/libero_mem_long_train
futuremamba/libero_mem_long_val
```

每帧保存训练所需 RGB、腕部 RGB、8D state、7D action 和由 LeRobot 维护的 `task_index`；metadata 保存 `tasks` 映射、`suite_id`、`task_id`、原始 `episode_id`、`task_family` 和 `memory_length` 供分层评测。两套 repo 都写入 split manifest、源数据 checksum 和固定 fork commit。

- [ ] **步骤 5：运行转换测试和 2-episode smoke**

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest -q \
  examples/libero_mem/convert_to_lerobot_test.py examples/libero_mem/env_adapter_test.py
uv run --group rlds examples/libero_mem/convert_to_lerobot.py \
  --libero-mem-data-dir "$LIBERO_MEM_RLDS" \
  --libero-long-data-dir "$LIBERO_LONG_RLDS" \
  --max-episodes-per-suite 2
```

预期：smoke repo 可由 `LeRobotEpisodeDataset` 读取一个 batch；episode/action 边界不交叉；split manifest 可重复生成。2-episode smoke 仅验证转换路径，不套用 100/20 与 40/10 数量断言。

- [ ] **步骤 6：提交**

```bash
git add .gitmodules third_party/libero examples/libero/Dockerfile \
  examples/libero/requirements.in examples/libero/requirements.txt \
  examples/libero_mem/convert_to_lerobot.py examples/libero_mem/convert_to_lerobot_test.py \
  examples/libero_mem/env_adapter.py examples/libero_mem/env_adapter_test.py
git commit -m "feat: integrate pinned LIBERO-Mem environment and data"
```

---
### 任务 10：实现闭环指标与结构化日志

**文件：**
- 创建：`examples/libero_mem/metrics.py`
- 创建：`examples/libero_mem/metrics_test.py`
- 创建：`examples/libero_mem/main.py`

- [ ] **步骤 1：为指标和符号事件写行为测试**

覆盖：Sequence 子目标只能按顺序首次计数；Or 只保留与已完成前缀兼容的分支；同一原子谓词从假变真并稳定 6 帧才产生一次事件；持续为真不能重复计数；已无待完成同签名子目标时再次触发计入 redundant execution；官方 `_overshot` 计入 overshoot；Wilson 95% 区间在 0/N 和 N/N 时有限；按 `task_family/memory_length` 聚合不会混淆 seed。

- [ ] **步骤 2：实现可复算的符号事件与指标累计器**

```python
@dataclasses.dataclass(frozen=True)
class EpisodeMetrics:
    success: bool
    completed_subgoals: int
    total_subgoals: int
    redundant_chunks: int
    decidable_chunks: int
    overshot: bool
    steps: int
```

`SymbolicEventMonitor` 从 `env.get_all_goals()` 展平每条可行 Sequence/Or 路径，以原子谓词签名和连续 6 帧真值产生 rising event。event 所在 query 若增加官方 satisfied-subgoal 前缀则为有效进展；若谓词签名已完成且所有兼容路径中都无该签名的待完成 occurrence，则该 query 为 redundant；其余不可判定动作不进入分母。日志保存每个 event 的谓词签名、frame、query、官方 progress before/after 和判定原因，使 Redundant Execution Rate 可从 JSONL 独立复算。

- [ ] **步骤 3：实现 LIBERO-Mem / LIBERO-Long 统一 runner**

每个 episode：创建/reset env → `LiberoMemEnvAdapter.reset()` → reset client →等待物体稳定→每 5 步重新 query→发送实际 executed prefix→每个 env step 后调用一次 `adapter.advance()`→写一条 JSONL。LIBERO-Long 不调用不存在的进度接口，只记录 success/steps。每条日志含 config、checkpoint checksum、task、task family、memory length、train seed、rollout seed、episode、success、完整 subgoal/event trace、redundant count、handoff ratio、history condition、timing 和视频路径。

任务级与总表从 JSONL 重新聚合，不从终端日志解析。`--task-suite-name` 支持 `libero_mem` 与 `libero_10`；两者使用各自 max steps。聚合器报告每任务固定 trial 数、训练 seed 均值、bootstrap 95% 区间、二项指标 Wilson 区间、随 `memory_length` 的 Temporal Scaling 曲线，以及 FutureMamba 相对同 seed 冻结基座的 LIBERO-Long Capability Retention 绝对差。

- [ ] **步骤 4：运行指标测试**

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest -q \
  examples/libero_mem/env_adapter_test.py examples/libero_mem/metrics_test.py
```

- [ ] **步骤 5：运行单任务单 episode smoke rollout**

启动 checkpoint server 后运行：

```bash
uv run examples/libero_mem/main.py \
  --task-suite-name libero_mem \
  --task-ids 0 \
  --num-trials-per-task 1 \
  --replan-steps 5 \
  --results-path data/libero_mem/smoke.jsonl
```

预期：进程正常退出；JSONL 恰有 1 条 episode；server 日志显示 episode reset；每个物理 step 只有一次增量进度检查；无论任务成功与否，actions、subgoal/event trace 和 timing 均完整。

- [ ] **步骤 6：提交**

```bash
git add examples/libero_mem/metrics.py examples/libero_mem/metrics_test.py examples/libero_mem/main.py
git commit -m "feat: evaluate progress and redundant execution"
```
---

### 任务 11：实现历史配对因果实验

**文件：**
- 创建：`examples/libero_mem/build_history_pairs.py`
- 创建：`examples/libero_mem/eval_history_pairs.py`
- 创建：`examples/libero_mem/history_pairs_test.py`

- [ ] **步骤 1：测试配对筛选和状态控制合同**

构造 synthetic query A/B/C/D：A/B 属于同一 task、进程不同、检索 Token 相似且目标动作分支不同；C 当前 Token 距离过远；D 下一谓词相同。断言只选择 A/B，并验证非重叠 episode、canonical physical-state checksum、唯一 current-observation checksum、A/B evaluator progress-state checksum和固定 noise seed 都是必填字段。

- [ ] **步骤 2：实现自动配对清单**

先按同 task、不同 progress label 和非重叠 episode 检索候选；当前 Token 余弦相似度只用于检索，不能替代相同输入控制。分别从初始 MuJoCo state replay A/B 历史，并保存 Policy state 与环境隐藏进程状态。选择 A 的物理 MuJoCo state（qpos/qvel/object state）作为 canonical physical state；A/B 最终推理都从该物理状态渲染同一 RGB、本体状态和指令，但短程评分时分别恢复各自的 `_satisfied_subgoals`、live/nonlive counters 与 `_overshot`。环境隐藏字段只能供 evaluator 使用，绝不输入模型。只保留 canonical physical state 下两个下一子目标谓词都可执行、且对应专家前 5 步动作语义不同的 pair：

```text
same task
history/progress label different
non-overlapping source episode
cosine(retrieval_token_a, retrieval_token_b) >= threshold
next-subgoal predicate a != next-subgoal predicate b
both predicates feasible in canonical physical state
```

按当前 Token 相似度降序贪心去重并输出 JSONL；每条含 episode/query ID、两段完整历史索引、canonical physical-state checksum、唯一 current-observation checksum、A/B evaluator-only progress-state checksum、两个目标谓词/专家分支和阈值。对象/subgoal annotation 只用于筛选与评分，不输入模型。

- [ ] **步骤 3：实现固定当前输入和固定噪声的历史干预**

本地加载 `FutureMambaPolicy`，分别 replay A/B 历史，获得当前 query 之前的 $S^A/S^B$ 与 evaluator-only 环境进程快照。每次条件推理前恢复对应 Policy 快照，固定 canonical observation 与 noise；短程执行前恢复相同 canonical physical state 和该 trial 对应的隐藏进程快照：

1. frozen baseline（零 state）；
2. 正确 state；
3. 清零、最近 $k$ 条截断、同 episode 确定性打乱 state；
4. 交换 $S^A/S^B$，但保留当前 trial 的 evaluator progress label；
5. 用首个执行 prefix 的短程 simulator event 判断进入目标谓词 A/B，并辅报到专家 prefix 的归一化距离。

结果必须验证所有条件的 current-observation / physical-state checksum 与 noise seed 完全相同，并记录 evaluator progress-state checksum。不能用两张不同当前图像分别推理；不能把隐藏进程标签输入模型；不能只报告 action L2 而没有语义分支判定。

- [ ] **步骤 4：运行因果实验测试**

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest -q examples/libero_mem/history_pairs_test.py
```

- [ ] **步骤 5：生成小型 pair manifest 并 smoke evaluate**

```bash
uv run examples/libero_mem/build_history_pairs.py \
  --repo-id futuremamba/libero_mem_long_val \
  --max-pairs-per-task 2 \
  --output data/libero_mem/history_pairs_smoke.jsonl
uv run examples/libero_mem/eval_history_pairs.py \
  --pairs data/libero_mem/history_pairs_smoke.jsonl \
  --results data/libero_mem/history_pairs_results_smoke.jsonl
```

预期：每个 result 包含 baseline、正确 state、清零/截断/打乱 state 和交换 state 条件；current observation、physical state 与 noise seed checksum 一致；分支判定可从 event trace 重算。

- [ ] **步骤 6：提交**

```bash
git add examples/libero_mem/build_history_pairs.py \
  examples/libero_mem/eval_history_pairs.py examples/libero_mem/history_pairs_test.py
git commit -m "feat: add history-conditioned branch evaluation"
```

---

### 任务 12：实现基线、机制消融与效率统计

**文件：**
- 修改：`src/openpi/models/futuremamba_config.py`
- 修改：`src/openpi/models/futuremamba.py`
- 修改：`src/openpi/models/futuremamba_test.py`
- 创建：`scripts/profile_futuremamba.py`
- 创建：`examples/libero_mem/run_experiment_matrix.py`
- 创建：`examples/libero_mem/run_experiment_matrix_test.py`

- [ ] **步骤 1：实现统一且参数可核验的 memory backend**

在任务 4 的 `FutureMambaConfig` 字段基础上实现 `mamba`、`gru`、`lstm`、`frame_stack`、`none`，均提供 `initial_state/step/scan`。新增统一 `MemoryState` PyTree 联合类型；`initial_memory_state`、`sample_actions_with_memory`、Policy snapshot/restore 和 profile 全部迁移到 `MemoryState`。GRU/LSTM/Frame Stack 隐宽通过整数搜索匹配 Mamba 可训练参数；误差超过 5% 时矩阵拒绝标记 `parameter_matched=true`。

- [ ] **步骤 2：实现主方法以外的 decoder/coupling 消融**

- `decoder_mode=handoff, coupling=hard`：前 K 步只用 Progress Expert；
- `coupling=convex`：前 K 步 `alpha_i*v_progress+(1-alpha_i)*v_action`；
- `coupling=residual`：前 K 步 `v_action+alpha_i*delta_v_progress`；
- `decoder_mode=action_memory_full`：不运行 Progress Expert，为冻结 Action Expert 的全部层追加由 memory 生成的 K/V，并在全部 N 步条件化；
- K 之后的 handoff 模式均只用 Action Expert；
- `bptt_window_queries=None` 为完整 episode 反传；整数值在窗口边界对 state `stop_gradient`，但用窗口前真实历史做无梯度 burn-in。

测试每种模式的 Expert 调用次数、解析速度值、Memory K/V 隔离和窗口梯度。

- [ ] **步骤 3：补齐当前条件、历史、表示和容量控制开关**

支持：`handoff_ratio={0,.1,.2,.3,.5,1}`、Progress 深度候选、`handoff_loss_weight=0`、`boundary_loss_weight=0`、`memory_input=token_only`、`conditioning_pool={attention,tokens4,tokens8}`、`use_prefix_cache=False`、`reset_memory_every_query`、`memory_backend`、`decoder_mode`、`frame_stack_window`、`bptt_window_queries`。对象 mask 只用于 oracle progress/object-aware 上限。所有开关进入 checkpoint metadata 和 JSONL。

- [ ] **步骤 4：实现预注册实验矩阵**

`run_experiment_matrix.py` 最低包含：Frozen task-adapted $\pi_{0.5}$、$\rho=0$ 数值等价、Recent Frame Stack、GRU、LSTM、Progress Expert + shared prefix KV without Memory、Progress Expert + Memory without prefix KV、Action Expert + Memory full horizon、Mamba Reset Every Query、Shuffled/Truncated/Zero History、Full-Horizon Progress Expert、oracle progress、两个交接损失、4 种 pooling/token 表示、depth、BPTT 和 3 种 coupling。主表固定 3 个训练 seed 和每任务相同 rollout seed/trial 数；配置启动前完整写入 JSON。

- [ ] **步骤 5：实现 profile 脚本**

固定 batch=1、10-step solver，报告：总参数、可训练参数、插件占比、Progress Expert FLOPs、Mamba state bytes、warmup 后 P50/P95 query latency、峰值 GPU memory和各 backend 参数匹配误差。若可训练参数占比超过 10%，输出 `lightweight_claim=false`。`--checkpoint-root` 接收包含 Orbax 数值 step 的实验根目录，并复用 `LatestCheckpointWeightLoader` 解析最大 step；不假设存在 `latest` 软链。

- [ ] **步骤 6：运行测试和 profile smoke**

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest -q \
  src/openpi/models/futuremamba_test.py examples/libero_mem/run_experiment_matrix_test.py
uv run scripts/profile_futuremamba.py \
  --config futuremamba_libero_mem \
  --checkpoint-root checkpoints/futuremamba_libero_mem/smoke
```

- [ ] **步骤 7：提交**

```bash
git add src/openpi/models/futuremamba_config.py src/openpi/models/futuremamba.py \
  src/openpi/models/futuremamba_test.py scripts/profile_futuremamba.py \
  examples/libero_mem/run_experiment_matrix.py examples/libero_mem/run_experiment_matrix_test.py
git commit -m "feat: add FutureMamba baselines ablations and profiling"
```

---

### 任务 13：端到端仿真验收

**文件：**
- 不新增功能文件
- 验证：上述模型、训练、策略和评测文件

- [ ] **步骤 1：运行全部离线相关测试**

```bash
env -i HOME="$HOME" PATH="$PWD/.venv/bin:/usr/bin" \
  PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=src \
  .venv/bin/pytest -q \
  src/openpi/models/mamba_test.py \
  src/openpi/models/progress_expert_test.py \
  src/openpi/models/futuremamba_test.py \
  src/openpi/models/pi0_test.py \
  src/openpi/training/episode_data_loader_test.py \
  src/openpi/training/weight_loaders_test.py \
  src/openpi/policies/futuremamba_policy_test.py \
  src/openpi/policies/libero_policy_test.py \
  src/openpi/transforms_test.py \
  src/openpi/serving/websocket_policy_server_test.py \
  packages/openpi-client/src/openpi_client/websocket_client_policy_test.py \
  scripts/train_futuremamba_test.py \
  examples/libero_mem/convert_to_lerobot_test.py \
  examples/libero_mem/env_adapter_test.py \
  examples/libero_mem/metrics_test.py \
  examples/libero_mem/history_pairs_test.py \
  examples/libero_mem/run_experiment_matrix_test.py
```

预期：0 failed。

- [ ] **步骤 2：运行插件训练 smoke**

```bash
JAX_PLATFORMS=cpu uv run scripts/train_futuremamba.py \
  futuremamba_libero_mem \
  --exp-name smoke \
  --batch-size 2 \
  --num-train-steps 2 \
  --overwrite
```

预期：产生可恢复 checkpoint；trainable path 全部位于 `futuremamba/`；frozen checksum 前后相同；loss 有限。

- [ ] **步骤 3：验证 $\rho=0$ 基座等价性**

用官方 task-adapted checkpoint、固定 observation 和 noise 比较 Frozen $\pi_{0.5}$ 与 FutureMamba($\rho=0$)。预期动作逐元素 `rtol=1e-6, atol=1e-6` 一致。

- [ ] **步骤 4：运行 1-episode 真正闭环 smoke**

启动 FutureMamba server，运行 LIBERO-Mem task 0 单 episode。验收点：reset RPC 生效；query 2 收到前 5 个实际 executed actions；Mamba state bytes 固定；前 K 步调用 Progress Expert、其余调用 Action Expert；JSONL 和视频生成。

- [ ] **步骤 5：运行最小因果 smoke**

至少 1 对 history pair 在同当前输入和同 noise 下完成正确 state 与交换 state 推理；结果文件可重算 Branch Accuracy。此步骤不要求模型已经取得正增益，只验证实验协议真实执行。

- [ ] **步骤 6：检查工作树与提交记录**

```bash
git status --short
git log --oneline --decorate -15
```

预期：工作树无未提交源码；每个任务有独立提交；没有数据、checkpoint、视频或 `.venv` 被跟踪。

---

## 近期证据与正式实验执行顺序

不要等所有软件模块完成后才验证环境。每个门槛失败都先修根因，再进入下一项：

1. **基座证据：** 完成任务 1 后，立即运行官方 task-adapted $\pi_{0.5}$ 的单任务单 episode LIBERO rollout，保存命令、checkpoint、seed、退出码、视频和原始日志；成功与否都必须证明闭环实际运行。
2. **数据证据：** 在实现模型主体期间并行完成 LIBERO-Mem 固定源码导入、任务列表枚举、官方顺序子目标 API smoke，以及 2-episode dataloader 读取；不能只展示下载完成。
3. **采样数值证据：** 任务 6 后立即验证 $K=0$ 与原 sampler 在同 observation/noise 下 `rtol=1e-6, atol=1e-6` 一致；构造同权重/同速度 fake expert 验证切换前后连续；记录 $K\in\{1,2,3,5\}$ 的调用次数与边界误差。
4. **训练证据：** 任务 7 后立即跑 2-step smoke，证明 loss 有限、插件参数变化、冻结参数 checksum 不变、checkpoint 可恢复。
5. **闭环 pilot：** 任务 10 后先在 1–2 个 LIBERO-Mem 任务运行 Frozen base 与完整 FutureMamba 单 seed，确保 Task Success、Subgoal Completion、Redundant Execution 和失败分类均可从 JSONL 重算。
6. **因果检查：** 清零、截断、打乱、交换历史的固定当前输入实验必须先于大规模 sweep；若正确 state 不改善 Branch Accuracy，停止扩大训练规模并检查记忆表示/数据对齐。
7. **必要基线：** 依次运行 Recent Frame Stack、无记忆 Progress Expert、无 prefix KV、Action Expert 全程记忆、参数匹配 GRU/LSTM 和随机未训练 Expert。
8. **机制消融：** 报告 $K\in\{0,1,2,3,5,10\}$、两个交接损失、Progress 深度、单/多 Token 表示、完整/截断 BPTT 与 hard/convex/residual。
9. **主表：** 只有预注册验证设置通过后，才运行 3 个训练 seed、固定 rollout trials、LIBERO-Long retention 和真机实验。

只有历史破坏导致收益消失、正确 state 提高 Branch Accuracy、且中间 $0<K<N$ 优于 $K=0/N$ 与 Action Expert 全程记忆时，论文才能把增益归因于任务进程记忆与噪声时间分工。
