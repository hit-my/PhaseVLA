# FutureMamba 纯 PyTorch 与官方 Mamba-2/3 迁移实现计划

> **面向 AI 代理的工作者：** 必需子技能：使用 superpowers:subagent-driven-development（推荐）或 superpowers:executing-plans 逐任务实现此计划。步骤使用复选框（`- [ ]`）语法跟踪进度。

**目标：** 将现有 FutureMamba 从 JAX/Flax 迁移为端到端纯 PyTorch 实现，在冻结的任务适配版 $\pi_{0.5}$ 上接入官方 Mamba-2 记忆、轻量 Progress Expert 高噪声硬交接，并在 RTX 5090 通过硬门后接入官方 Mamba-3 SISO。

**架构：** `FutureMambaPytorch` 持有冻结且始终处于 `eval()` 的 `PI0Pytorch` 基座，以及唯一可训练的 `FutureMambaPluginPytorch`。一次冻结 prefix forward 同时产生最后层 hidden、pad mask 和逐层 KV；Mamba 按 action-chunk query 递推固定大小状态，Progress Expert 只在前 $K=\lceil\rho N\rceil$ 个去噪步读取 Memory Token，原 Action Expert 从 $x_K$ 接管。Mamba-2 与 Mamba-3 来自同一个官方 `v2.3.2` 提交，但参数、状态和 checkpoint 严格隔离。

**技术栈：** Python 3.11、PyTorch 2.9.1、Triton 3.5.1、Transformers 4.53.2、官方 `state-spaces/mamba` v2.3.2、SafeTensors、LeRobot、pytest、WebSocket/msgpack。

**工作区：** `/home/ubuntu/.config/superpowers/worktrees/openpi/futuremamba-pytorch-migration-plan`，分支 `docs/futuremamba-pytorch-migration-plan`，计划基线 `cfde3b2`。

**设计规格：** `docs/superpowers/specs/2026-08-10-futuremamba-pytorch-mamba-migration-design.md`。

**基座 checkpoint：** `/home/ubuntu/lgd/CoRL2026/checkpoints/pi05_libero_libero_pi05_2000`，其中 `params/` 是 Orbax/JAX 参数，`assets/` 位于 checkpoint 根目录内。

---

## 固定接口与文件结构

### 新建

- `environments/futuremamba/pyproject.toml`：隔离依赖门与最终版本来源。
- `environments/futuremamba/check_runtime.py`：Torch、Triton、CUDA、GPU、官方 commit 与 kernel 导入检查。
- `src/openpi/models_pytorch/futuremamba_config.py`：纯 PyTorch 配置、精确 backend 名称、checkpoint metadata。
- `src/openpi/models_pytorch/mamba_memory.py`：统一 `MemoryBackend`、Mamba-2、Mamba-3 SISO 及消融后端。
- `src/openpi/models_pytorch/mamba_memory_test.py`：sequence/step、mask、reset、snapshot、schema 与硬件门测试。
- `src/openpi/models_pytorch/progress_expert.py`：只读 prefix KV + 单 Memory Token 的轻量 Gemma expert。
- `src/openpi/models_pytorch/progress_expert_test.py`：层映射、mask、cache 不变性和梯度测试。
- `src/openpi/models_pytorch/futuremamba.py`：冻结基座、插件、episode loss 与硬交接采样。
- `src/openpi/models_pytorch/futuremamba_test.py`：冻结、记忆输入、损失、硬交接和因果测试。
- `src/openpi/training/futuremamba_conditioning_cache.py`：确定性离线 prefix conditioning cache。
- `src/openpi/training/futuremamba_conditioning_cache_test.py`：checksum、随机增强拒绝与完整 KV 测试。
- `src/openpi/training/futuremamba_checkpoint.py`：仅插件 checkpoint 的严格保存与恢复。
- `src/openpi/training/futuremamba_checkpoint_test.py`：schema、backend、base checksum 与恢复测试。
- `scripts/precompute_futuremamba_prefix.py`：按 episode/query 预计算冻结 conditioning。
- `scripts/train_futuremamba_pytorch.py`：完整 episode PyTorch BPTT 训练入口。
- `scripts/train_futuremamba_pytorch_test.py`：2-step 更新、保存、恢复到第 4 step。
- `scripts/validate_jax_pytorch_pi05.py`：JAX/PyTorch 基座逐层与 10-step 数值验收。
- `scripts/check_mamba3_rtx5090.py`：Mamba-3 十项硬门与机器可读结果。

### 修改

- `pyproject.toml`、`uv.lock`：仅在隔离门通过后统一到已验证的 Torch/Triton/Mamba 组合。
- `examples/convert_jax_model_to_pytorch.py`：LoRA 扫描、严格 key 校验、正确复制 assets、转换 manifest。
- `src/openpi/models/model.py`：让 PyTorch 配置自行构造模型，不再硬编码 `PI0Pytorch`。
- `src/openpi/models_pytorch/gemma_pytorch.py`：稳定的只读 prefix KV view 与 Progress layer 原语。
- `src/openpi/models_pytorch/pi0_pytorch.py`：`FrozenPrefix`、单次 prefix 编码与冻结 Action Expert 单步接口。
- `src/openpi/training/config.py`：注册纯 PyTorch FutureMamba Stage B 配置。
- `src/openpi/training/episode_data_loader.py`：PyTorch episode batch 转换，不改变数据语义。
- `src/openpi/policies/futuremamba_policy.py`：从 JAX state 改为显式 PyTorch state。
- `src/openpi/policies/futuremamba_policy_test.py`：backend/schema、executed-action、reset/snapshot/restore。
- `src/openpi/policies/policy_config.py`：显式识别 `plugin.safetensors` bundle 并严格加载。
- `src/openpi/serving/websocket_policy_server_test.py`：真实 FutureMamba state 的连接隔离与 reset ack。
- `examples/libero_mem/main.py`、`examples/libero_mem/eval_history_pairs.py`：新 bundle 与 state schema。
- `examples/libero_mem/run_experiment_matrix.py`：`mamba2` / `mamba3_siso` 命名及后端代际消融。
- `examples/libero_mem/run_experiment_matrix_test.py`：矩阵与 metadata 约束。
- `scripts/profile_futuremamba.py`：PyTorch 参数、state bytes、延迟、显存与 kernel 模式。

### 最终删除

- `src/openpi/models/mamba.py`、`src/openpi/models/mamba_test.py`。
- `src/openpi/models/progress_expert.py`、`src/openpi/models/progress_expert_test.py`。
- `src/openpi/models/futuremamba.py`、`src/openpi/models/futuremamba_test.py`。
- `src/openpi/models/futuremamba_config.py`。
- `scripts/train_futuremamba.py`、`scripts/train_futuremamba_test.py`。
- 只服务上述 JAX 实现的 import、配置分支和 `memory_backend="mamba"`。

### 类型合同

所有任务使用以下名称，不得另起同义接口：

```python
@dataclasses.dataclass(frozen=True)
class FrozenPrefix:
    hidden: torch.Tensor
    pad_mask: torch.BoolTensor
    kv_cache: object


@dataclasses.dataclass(frozen=True)
class MemorySnapshot:
    backend_id: str
    state_schema_version: int
    batch_size: int
    layers: tuple[tuple[torch.Tensor, ...], ...]


class MemoryBackend(nn.Module):
    backend_id: str
    state_schema_version: int

    def initial_state(self, batch_size: int, *, device: torch.device, dtype: torch.dtype) -> MemorySnapshot: ...
    def forward_sequence(
        self, x: torch.Tensor, *, query_mask: torch.BoolTensor | None = None
    ) -> tuple[torch.Tensor, MemorySnapshot]: ...
    @torch.no_grad()
    def step(self, x: torch.Tensor, state: MemorySnapshot) -> tuple[torch.Tensor, MemorySnapshot]: ...
    def reset(self, state: MemorySnapshot, mask: torch.BoolTensor) -> MemorySnapshot: ...
    def snapshot(self, state: MemorySnapshot) -> MemorySnapshot: ...
    def restore(self, snapshot: MemorySnapshot, *, batch_size: int, device: torch.device) -> MemorySnapshot: ...
```

`forward_sequence` 只接受右侧 padding；`step` 的输入/输出均为 `[B,D]`。`MemorySnapshot.layers` 对 Mamba-2 每层含 `(conv_state, ssm_state)`，对 Mamba-3 SISO 每层含 `(angle_dt_state, ssm_state, k_state, v_state)`。

---

## 阶段 A：依赖与基座迁移

### 任务 1：建立官方 Mamba 隔离依赖门

**文件：**
- 创建：`environments/futuremamba/pyproject.toml`
- 创建：`environments/futuremamba/uv.lock`
- 创建：`environments/futuremamba/check_runtime.py`
- 验证：官方 Mamba `v2.3.2` 提交 `77069de5cdb55cbe98b670889c80df211e031039`

- [ ] **步骤 1：写入隔离项目与精确依赖**

```toml
[project]
name = "openpi-futuremamba-runtime"
version = "0.0.0"
requires-python = ">=3.11,<3.12"
dependencies = [
  "openpi",
  "openpi-client",
  "torch==2.9.1",
  "triton==3.5.1",
  "mamba-ssm",
  "pytest>=8.3.4",
]

[tool.uv.sources]
openpi = { path = "../..", editable = true }
openpi-client = { path = "../../packages/openpi-client", editable = true }
lerobot = { git = "https://github.com/huggingface/lerobot", rev = "0cf864870cf29f4738d3ade893e6fd13fbd7cdb5" }
mamba-ssm = { git = "https://github.com/state-spaces/mamba", rev = "77069de5cdb55cbe98b670889c80df211e031039" }

[tool.uv.extra-build-dependencies]
mamba-ssm = [{ requirement = "torch==2.9.1", match-runtime = true }]

[tool.uv]
package = false
override-dependencies = [
  "torch==2.9.1",
  "triton==3.5.1",
  "ml-dtypes==0.4.1",
  "tensorstore==0.1.74",
]
```

- [ ] **步骤 2：编写失败的 runtime probe**

`check_runtime.py` 必须导入 `Mamba2`、`Mamba3`、`mamba3_siso_combined`、`apply_rotary_qk_inference_fwd` 和 `mamba3_step_fn`，并输出单行 JSON：

```python
required = {
    "torch": torch.__version__,
    "triton": triton.__version__,
    "cuda": torch.version.cuda,
    "gpu": torch.cuda.get_device_name(0),
    "compute_capability": list(torch.cuda.get_device_capability(0)),
    "mamba2": Mamba2 is not None,
    "mamba3": Mamba3 is not None,
    "mamba3_sequence": mamba3_siso_combined is not None,
    "mamba3_rotary_step": apply_rotary_qk_inference_fwd is not None,
    "mamba3_cute_step": mamba3_step_fn is not None,
}
print(json.dumps(required, sort_keys=True))
```

没有 CUDA、Torch/Triton 版本不匹配或 Mamba-2 必需 import 缺失时退出码必须非 0。Mamba-3 类与 kernel 的 import 结果只写入 JSON，是否允许正式使用由任务 14 的十项硬门决定；不能因为 Mamba-3 缺失就伪装成可用，也不能阻塞 Mamba-2 基线。

- [ ] **步骤 3：运行 probe，确认当前环境失败而非静默通过**

```bash
python environments/futuremamba/check_runtime.py
```

预期：当前主环境因缺少已锁定的 Torch/Mamba 栈退出非 0；输出明确缺失项。

- [ ] **步骤 4：用 CUDA 12.8 源码构建隔离环境并运行依赖门**

默认 `nvcc` 已观察为 CUDA 11.5；它低于官方 Mamba 最低版本且不能生成 RTX 5090 的 `sm_120`。本任务必须显式选择机器上的 `/usr/local/cuda-12.8`：

```bash
env CUDA_HOME=/usr/local/cuda-12.8 \
  PATH="/usr/local/cuda-12.8/bin:$PATH" \
  LD_LIBRARY_PATH="/usr/local/cuda-12.8/lib64:${LD_LIBRARY_PATH:-}" \
  MAMBA_FORCE_BUILD=TRUE \
  TORCH_CUDA_ARCH_LIST=12.0 \
  UV_PROJECT_ENVIRONMENT="$PWD/.venv-futuremamba" \
  uv sync --project environments/futuremamba
env CUDA_HOME=/usr/local/cuda-12.8 \
  PATH="/usr/local/cuda-12.8/bin:$PATH" \
  LD_LIBRARY_PATH="/usr/local/cuda-12.8/lib64:${LD_LIBRARY_PATH:-}" \
  UV_PROJECT_ENVIRONMENT="$PWD/.venv-futuremamba" \
  uv run --project environments/futuremamba python environments/futuremamba/check_runtime.py
```

预期：build log 显示 CUDA 12.8 与 `sm_120`，Torch `2.9.1`、Triton `3.5.1`，RTX 5090 compute capability 为 `[12, 0]`，Mamba-2 import 和最小 forward 均成功。Mamba-3 类与各 kernel import 分项记录；任一项失败时不修改官方源码，也不谎报 Mamba-3 可用。

- [ ] **步骤 5：提交隔离门**

```bash
git add environments/futuremamba
git commit -m "chore(依赖): 添加 FutureMamba 官方运行时门"
```

---

### 任务 2：严格转换任务适配版 $\pi_{0.5}$ checkpoint

**文件：**
- 修改：`examples/convert_jax_model_to_pytorch.py`
- 创建：`examples/convert_jax_model_to_pytorch_test.py`
- 创建：`scripts/validate_jax_pytorch_pi05.py`

- [ ] **步骤 1：编写失败测试，锁定 LoRA、key 与 assets 合同**

```python
def test_conversion_rejects_lora_without_merge():
    flat = {"PaliGemma/llm/lora_a": np.ones((2, 2))}
    with pytest.raises(ValueError, match="LoRA"):
        validate_convertible_parameter_tree(flat)


def test_strict_load_only_allows_verified_tied_lm_head():
    validate_load_result(
        missing=["paligemma_with_expert.paligemma.language_model.lm_head.weight"],
        unexpected=[],
        tied_weight_verified=True,
    )
    with pytest.raises(ValueError, match="unexpected"):
        validate_load_result(missing=[], unexpected=["bad.weight"], tied_weight_verified=True)


def test_assets_are_copied_from_checkpoint_root(tmp_path):
    source = tmp_path / "checkpoint"
    (source / "assets" / "physical-intelligence").mkdir(parents=True)
    (source / "assets" / "physical-intelligence" / "norm_stats.json").write_text("{}")
    copy_checkpoint_assets(source, tmp_path / "converted")
    assert (tmp_path / "converted/assets/physical-intelligence/norm_stats.json").read_text() == "{}"
```

- [ ] **步骤 2：运行测试验证失败**

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=src \
  .venv-futuremamba/bin/pytest -q examples/convert_jax_model_to_pytorch_test.py
```

预期：FAIL；缺少三个严格转换 helper。

- [ ] **步骤 3：实现严格转换与转换 manifest**

转换流程必须：

```python
flat_paths = tuple("/".join(path) for path in traversals.flatten_mapping(params).keys())
if any("lora_a" in path or "lora_b" in path for path in flat_paths):
    raise ValueError("LoRA parameters detected; merge adapters with a verified converter before conversion")

incompatible = pi0_model.load_state_dict(all_params, strict=False)
allowed_missing = {"paligemma_with_expert.paligemma.language_model.lm_head.weight"}
if set(incompatible.missing_keys) != allowed_missing or incompatible.unexpected_keys:
    raise ValueError(
        f"strict conversion mismatch: missing={incompatible.missing_keys}, "
        f"unexpected={incompatible.unexpected_keys}"
    )
```

只有 `lm_head.weight` 与输入 embedding 的存储共享已通过 `data_ptr()` 校验时，才允许上述唯一 missing key。assets 来源固定为 `pathlib.Path(checkpoint_dir) / "assets"`。`conversion_manifest.json` 记录源参数 key/shape/dtype、源 checkpoint checksum、assets checksum、目标 key/shape/dtype 与转换配置。

- [ ] **步骤 4：运行转换单元测试**

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=src \
  .venv-futuremamba/bin/pytest -q examples/convert_jax_model_to_pytorch_test.py
```

预期：全部通过。

- [ ] **步骤 5：执行 float32 转换**

```bash
PYTHONPATH=src .venv-futuremamba/bin/python examples/convert_jax_model_to_pytorch.py \
  --checkpoint-dir /home/ubuntu/lgd/CoRL2026/checkpoints/pi05_libero_libero_pi05_2000 \
  --config-name pi05_libero \
  --output-path /home/ubuntu/lgd/CoRL2026/checkpoints/pi05_libero_libero_pi05_2000_pytorch \
  --precision float32
```

预期：输出目录含 `model.safetensors`、`config.json`、`conversion_manifest.json` 和完整 `assets/physical-intelligence/`；不存在未解释 missing/unexpected key。

- [ ] **步骤 6：实现并运行逐层 parity**

`validate_jax_pytorch_pi05.py` 固定 observation、action、noise、time 与 seed，比较：最后有效 prefix token、映射层 prefix K/V、Action Expert 单步速度和完整 10-step action chunk。命令：

```bash
JAX_PLATFORMS=cpu PYTHONPATH=src .venv-futuremamba/bin/python scripts/validate_jax_pytorch_pi05.py \
  --jax-checkpoint /home/ubuntu/lgd/CoRL2026/checkpoints/pi05_libero_libero_pi05_2000 \
  --pytorch-checkpoint /home/ubuntu/lgd/CoRL2026/checkpoints/pi05_libero_libero_pi05_2000_pytorch \
  --config pi05_libero --dtype float32 --seed 0
```

预期：每个比较项 `mean_absolute_error <= 1e-4` 且 `max_absolute_error <= 5e-4`。失败时脚本报告首个分歧层并退出非 0。

- [ ] **步骤 7：提交转换修复**

```bash
git add examples/convert_jax_model_to_pytorch.py \
  examples/convert_jax_model_to_pytorch_test.py scripts/validate_jax_pytorch_pi05.py
git commit -m "fix(模型转换): 严格迁移任务适配版 pi0.5 权重"
```

---

### 任务 3：定义纯 PyTorch 配置与模型工厂

**文件：**
- 创建：`src/openpi/models_pytorch/futuremamba_config.py`
- 创建：`src/openpi/models_pytorch/futuremamba_config_test.py`
- 修改：`src/openpi/models/model.py`

- [ ] **步骤 1：编写失败测试，锁定命名和配置不变量**

```python
def test_backend_names_are_explicit():
    assert FutureMambaPytorchConfig(memory_backend="mamba2").memory_backend == "mamba2"
    assert FutureMambaPytorchConfig(memory_backend="mamba3_siso").memory_backend == "mamba3_siso"
    with pytest.raises(ValueError, match="memory_backend"):
        FutureMambaPytorchConfig(memory_backend="mamba")


def test_default_capacity_and_handoff():
    config = FutureMambaPytorchConfig()
    assert dataclasses.asdict(config.memory) == {
        "d_model": 1024, "depth": 2, "d_state": 128, "expand": 2,
        "headdim": 64, "ngroups": 1, "d_conv": 4,
        "rms_norm": True, "residual_in_fp32": True,
        "fused_add_norm": False, "use_mem_eff_path": False,
    }
    assert config.progress_depth == 4
    assert config.executed_horizon == 5
    assert config.num_denoise_steps == 10
```

- [ ] **步骤 2：运行测试验证失败**

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=src \
  .venv-futuremamba/bin/pytest -q src/openpi/models_pytorch/futuremamba_config_test.py
```

预期：FAIL；模块不存在。

- [ ] **步骤 3：实现配置**

定义 `MambaMemoryConfig` 和 `FutureMambaPytorchConfig(pi0_config.Pi0Config)`。后者固定 `pi05=True`、`discrete_state_input=True`，支持 `mamba2`、`mamba3_siso`、`gru`、`lstm`、`frame_stack`、`none`，并验证 `0 <= handoff_ratio <= 1`、`executed_horizon <= action_horizon`、`progress_depth <= action expert depth`。`checkpoint_metadata()` 返回设计规格第 10.1 节全部字段，不使用 `backend="mamba"`。

配置实现以下显式 PyTorch 工厂：

```python
def create_pytorch(self) -> "FutureMambaPytorch":
    from openpi.models_pytorch.futuremamba import FutureMambaPytorch
    return FutureMambaPytorch(self)


def create(self, rng):
    del rng
    raise RuntimeError("FutureMambaPytorchConfig is PyTorch-only; call create_pytorch()")
```

`BaseModelConfig.load_pytorch()` 改为调用 `create_pytorch()`（若存在），普通 `Pi0Config` 保持原行为。

- [ ] **步骤 4：运行配置与普通 PI0 回归**

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=src \
  .venv-futuremamba/bin/pytest -q \
  src/openpi/models_pytorch/futuremamba_config_test.py \
  src/openpi/models/model_test.py
```

预期：全部通过，普通 `Pi0Config` 的模型类型和输入规格不变。

- [ ] **步骤 5：提交配置工厂**

```bash
git add src/openpi/models/model.py src/openpi/models_pytorch/futuremamba_config.py \
  src/openpi/models_pytorch/futuremamba_config_test.py
git commit -m "feat(配置): 添加纯 PyTorch FutureMamba 配置"
```

---

## 阶段 B：Mamba-2 与冻结 prefix

### 任务 4：提取无行为变化的冻结 prefix 与 Action Expert helper

**文件：**
- 修改：`src/openpi/models_pytorch/gemma_pytorch.py`
- 修改：`src/openpi/models_pytorch/pi0_pytorch.py`
- 创建：`src/openpi/models_pytorch/pi0_pytorch_test.py`

- [ ] **步骤 1：编写失败测试，锁定单次 prefix 与只读 cache**

```python
def test_last_valid_prefix_token_ignores_padding(tiny_pi0, observation):
    frozen = tiny_pi0.encode_frozen_prefix(observation, train=False)
    index = frozen.pad_mask.long().sum(dim=-1) - 1
    expected = frozen.hidden[torch.arange(index.numel()), index]
    torch.testing.assert_close(tiny_pi0.last_valid_prefix(frozen), expected)


def test_frozen_prefix_is_detached_normal_tensor(tiny_pi0, observation):
    frozen = tiny_pi0.encode_frozen_prefix(observation, train=False)
    assert frozen.hidden.requires_grad is False
    assert frozen.hidden.is_inference() is False
    assert all(t.requires_grad is False and not t.is_inference() for t in iter_cache_tensors(frozen.kv_cache))


def test_refactored_sampling_matches_original_fixed_noise(tiny_pi0, observation, noise):
    before = legacy_sample_actions(tiny_pi0, observation, noise=noise, num_steps=10)
    after = tiny_pi0.sample_actions("cpu", observation, noise=noise, num_steps=10)
    torch.testing.assert_close(after, before, rtol=0, atol=1e-6)
```

另加空 `pad_mask` 必须抛 `ValueError("prefix contains no valid token")`，以及 Progress cache clone 后原 cache checksum 不变的测试。

- [ ] **步骤 2：运行测试验证失败**

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=src \
  .venv-futuremamba/bin/pytest -q src/openpi/models_pytorch/pi0_pytorch_test.py
```

预期：FAIL；缺少 `FrozenPrefix` 与 helper。

- [ ] **步骤 3：实现固定 helper**

`pi0_pytorch.py` 增加：

```python
@dataclasses.dataclass(frozen=True)
class FrozenPrefix:
    hidden: torch.Tensor
    pad_mask: torch.BoolTensor
    kv_cache: object


@torch.no_grad()
def encode_frozen_prefix(self, observation, *, train: bool = False) -> FrozenPrefix:
    images, img_masks, lang_tokens, lang_masks, _ = self._preprocess_observation(observation, train=train)
    prefix_embs, pad_mask, att_mask = self.embed_prefix(images, img_masks, lang_tokens, lang_masks)
    mask_2d = make_att_2d_masks(pad_mask, att_mask)
    positions = torch.cumsum(pad_mask, dim=1) - 1
    (hidden, _), cache = self.paligemma_with_expert.forward(
        attention_mask=self._prepare_attention_masks_4d(mask_2d),
        position_ids=positions,
        past_key_values=None,
        inputs_embeds=[prefix_embs, None],
        use_cache=True,
    )
    return FrozenPrefix(hidden=hidden.detach(), pad_mask=pad_mask.detach(), kv_cache=detach_cache(cache))
```

禁止 `torch.inference_mode()`。`gemma_pytorch.py` 提供 `iter_cache_tensors()`、`detach_cache()` 和 `clone_selected_prefix_cache(cache, layer_indices)`；clone 后不共享可变 storage。

将现有 `denoise_step` 保留为公开冻结 Action Expert 单步 helper，`sample_actions` 改为调用 `encode_frozen_prefix`，不重复 VLM forward。

- [ ] **步骤 4：运行数值与 cache 回归**

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=src \
  .venv-futuremamba/bin/pytest -q src/openpi/models_pytorch/pi0_pytorch_test.py
```

预期：全部通过；固定噪声 10-step 输出逐元素相同，prefix forward 调用次数严格为 1。

- [ ] **步骤 5：提交 prefix 重构**

```bash
git add src/openpi/models_pytorch/gemma_pytorch.py \
  src/openpi/models_pytorch/pi0_pytorch.py src/openpi/models_pytorch/pi0_pytorch_test.py
git commit -m "refactor(PyTorch基座): 提取只读 prefix 缓存接口"
```

---

### 任务 5：实现官方 Mamba-2 MemoryBackend

**文件：**
- 创建：`src/openpi/models_pytorch/mamba_memory.py`
- 创建：`src/openpi/models_pytorch/mamba_memory_test.py`

- [ ] **步骤 1：编写失败测试，定义公共 state 合同**

```python
def test_mamba2_sequence_matches_steps():
    torch.manual_seed(0)
    model = Mamba2MemoryBackend(MambaMemoryConfig(d_model=32, depth=2, d_state=8, headdim=8))
    x = torch.randn(2, 7, 32)
    sequence_y, sequence_state = model.forward_sequence(x)
    state = model.initial_state(2, device=x.device, dtype=x.dtype)
    step_y = []
    for query in range(x.shape[1]):
        y, state = model.step(x[:, query], state)
        step_y.append(y)
    torch.testing.assert_close(sequence_y, torch.stack(step_y, dim=1), rtol=1e-4, atol=1e-4)
    assert_state_close(sequence_state, state, rtol=1e-4, atol=1e-4)


def test_padding_queries_do_not_advance_state():
    mask = torch.tensor([[True, True, False], [True, True, True]])
    y, state = model.forward_sequence(torch.randn(2, 3, 32), query_mask=mask)
    assert torch.count_nonzero(y[0, 2]) == 0
    expected = model.forward_sequence(x[:1, :2])[1]
    assert_state_row_close(state, 0, expected, 0)


def test_non_right_padding_is_rejected():
    with pytest.raises(ValueError, match="right padding"):
        model.forward_sequence(x, query_mask=torch.tensor([[True, False, True]]))
```

同时测试 partial batch reset、snapshot clone、backend/schema/shape/dtype/batch restore 拒绝，以及长度 2 与 2048 的 state tree/shape/bytes 完全相同。

- [ ] **步骤 2：运行测试验证失败**

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=src \
  .venv-futuremamba/bin/pytest -q src/openpi/models_pytorch/mamba_memory_test.py -k mamba2
```

预期：FAIL；模块不存在。

- [ ] **步骤 3：实现官方 Block stack**

每层使用官方 `Block`，mixer 为官方 `Mamba2`，`d_intermediate=0` 等价为 `nn.Identity`，`fused_add_norm=False`、`residual_in_fp32=True`。sequence 路径调用官方 mixer `forward`；step 路径只显式复现官方 Block 的 Add → RMSNorm → 官方 `mixer.step`，最后应用官方 RMSNorm。禁止复制 Mamba 状态方程。

Mamba-2 state 必须由 `layer.mixer.allocate_inference_cache()` 创建：

```python
layers = tuple(
    tuple(t.detach().clone() for t in block.mixer.allocate_inference_cache(batch_size, max_seqlen=1, dtype=dtype))
    for block in self.layers
)
return MemorySnapshot("mamba2", self.state_schema_version, batch_size, layers)
```

`forward_sequence` 按每行有效长度调用一次官方 causal sequence forward，再 padding 输出；零有效 query 直接拒绝。最终 state 由内部官方 inference cache clone 得到，禁止把 `InferenceParams` 暴露给调用方。

- [ ] **步骤 4：运行 Mamba-2 合同测试**

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=src \
  .venv-futuremamba/bin/pytest -q src/openpi/models_pytorch/mamba_memory_test.py -k mamba2
```

预期：全部通过；sequence/step tolerance 为 `rtol=atol=1e-4`，state bytes 不随序列长度变化。

- [ ] **步骤 5：实现参数匹配消融后端**

在同一文件实现 `GRUMemoryBackend`、`LSTMMemoryBackend`、`FrameStackMemoryBackend` 与 `NoMemoryBackend`。每个后端遵循同一 mask/reset/snapshot 合同，但使用各自 backend ID；隐藏宽度取使实际参数量最接近 Mamba-2 的整数值，并在 metadata 中写实际误差，误差大于 5% 时不得标记 `parameter_matched=True`。

- [ ] **步骤 6：运行所有 MemoryBackend 测试**

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=src \
  .venv-futuremamba/bin/pytest -q src/openpi/models_pytorch/mamba_memory_test.py
```

预期：全部通过。

- [ ] **步骤 7：提交 MemoryBackend**

```bash
git add src/openpi/models_pytorch/mamba_memory.py src/openpi/models_pytorch/mamba_memory_test.py
git commit -m "feat(记忆后端): 接入官方 Mamba-2 状态接口"
```

---

### 任务 6：实现只读 prefix 的 PyTorch Progress Expert

**文件：**
- 修改：`src/openpi/models_pytorch/gemma_pytorch.py`
- 创建：`src/openpi/models_pytorch/progress_expert.py`
- 创建：`src/openpi/models_pytorch/progress_expert_test.py`

- [ ] **步骤 1：编写失败测试，锁定 token 隔离与 cache 只读性**

```python
def test_layer_mapping_covers_first_and_last_layer():
    assert make_layer_mapping(action_depth=18, progress_depth=4) == (0, 6, 11, 17)


def test_memory_token_only_conditions_progress_queries(expert, prefix_cache):
    before = cache_checksums(prefix_cache)
    a = expert(prefix_cache, prefix_mask, torch.zeros(2, 1, 1024), noisy_actions, timestep)
    b = expert(prefix_cache, prefix_mask, torch.ones(2, 1, 1024), noisy_actions, timestep)
    assert not torch.allclose(a, b)
    assert cache_checksums(prefix_cache) == before


def test_memory_token_has_kv_but_no_query(expert, monkeypatch):
    seen = capture_projection_lengths(expert, monkeypatch)
    expert(prefix_cache, prefix_mask, memory_token, noisy_actions, timestep)
    assert seen["query_tokens"] == action_horizon
    assert seen["key_value_tokens"] == action_horizon + 1
```

还需测试 prefix padding 不可见、输出只含 `action_horizon`、Progress 参数有梯度、prefix cache 无梯度且未改变、Action Expert 参数无梯度。

- [ ] **步骤 2：运行测试验证失败**

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=src \
  .venv-futuremamba/bin/pytest -q src/openpi/models_pytorch/progress_expert_test.py
```

预期：FAIL；Progress Expert 不存在。

- [ ] **步骤 3：实现 CachedPrefixActionLayerPytorch**

该层复用 Transformers Gemma decoder layer 的 `input_layernorm`、Q/K/V/O、RoPE、`post_attention_layernorm` 和 MLP。每层输入为 `[Memory KV][Action QKV]`，prefix K/V 由对应 `layer_mapping[j]` 的只读 clone 提供。attention key/value 顺序固定为：

```text
[prefix K/V] [memory K/V] [action K/V]
```

query 只来自 action token；attention mask 形状固定 `[B,1,A,C+1+A]`。Memory Token 不进入 base VLM，也不传给 Action Expert。

- [ ] **步骤 4：实现 ProgressExpertPytorch**

```python
class ProgressExpertPytorch(nn.Module):
    def forward(
        self,
        prefix_cache: PrefixKVView,
        prefix_mask: torch.BoolTensor,
        memory_token: torch.Tensor,
        noisy_actions: torch.Tensor,
        timestep: torch.Tensor,
    ) -> torch.Tensor: ...
```

默认深度为 4，宽度/head/KV head/head dim/MLP/激活/AdaRMS 与 `gemma_300m` 一致。参数随机初始化；不得从 Action Expert 复制权重。最后只取 action token，经独立 `action_out_proj` 输出 `[B,H,A]` 速度。

- [ ] **步骤 5：运行 Progress Expert 测试**

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=src \
  .venv-futuremamba/bin/pytest -q src/openpi/models_pytorch/progress_expert_test.py
```

预期：全部通过，原始 prefix cache checksum 前后相同。

- [ ] **步骤 6：提交 Progress Expert**

```bash
git add src/openpi/models_pytorch/gemma_pytorch.py \
  src/openpi/models_pytorch/progress_expert.py src/openpi/models_pytorch/progress_expert_test.py
git commit -m "feat(进度专家): 实现只读 prefix 高噪声解码器"
```

---

## 阶段 C：端到端模型与训练

### 任务 7：实现 FutureMambaPytorch 记忆输入与硬交接

**文件：**
- 创建：`src/openpi/models_pytorch/futuremamba.py`
- 创建：`src/openpi/models_pytorch/futuremamba_test.py`

- [ ] **步骤 1：编写失败测试，锁定冻结与记忆输入**

```python
def test_only_plugin_parameters_are_trainable(model):
    trainable = {name for name, p in model.named_parameters() if p.requires_grad}
    assert trainable
    assert all(name.startswith("futuremamba.") for name in trainable)
    assert model.base.training is False


def test_executed_action_summary_uses_only_valid_prefix(plugin):
    actions = torch.tensor([[[1.0, 2.0], [3.0, 4.0], [99.0, 99.0]]])
    mask = torch.tensor([[True, True, False]])
    summary = plugin.summarize_executed_actions(actions, mask)
    torch.testing.assert_close(summary, torch.tensor([[2.0, 3.0, 3.0, 4.0]]))


def test_padding_query_neither_updates_memory_nor_contributes_loss(model, episode_batch):
    result = model.compute_episode_loss(episode_batch)
    episode_batch.actions[:, -1] = 1e6
    second = model.compute_episode_loss(episode_batch)
    torch.testing.assert_close(result["loss"], second["loss"])
```

- [ ] **步骤 2：编写失败测试，锁定硬交接调用次数**

```python
@pytest.mark.parametrize(("ratio", "progress_calls", "action_calls"), [
    (0.0, 0, 10), (0.2, 2, 8), (1.0, 10, 0),
])
def test_hard_handoff_call_counts(model, ratio, progress_calls, action_calls):
    _, _, diagnostics = model.sample_actions_with_memory(
        observation, model.initial_memory_state(1), executed_actions, executed_mask,
        noise=noise, num_steps=10, handoff_ratio=ratio,
    )
    assert diagnostics["progress_calls"] == progress_calls
    assert diagnostics["action_calls"] == action_calls
```

`ratio=0` 还必须与冻结 `PI0Pytorch.sample_actions` 在固定 noise 下逐元素等价；`ratio=1` 必须证明 Action Expert 未被调用。

- [ ] **步骤 3：运行测试验证失败**

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=src \
  .venv-futuremamba/bin/pytest -q src/openpi/models_pytorch/futuremamba_test.py
```

预期：FAIL；复合模型不存在。

- [ ] **步骤 4：实现插件与严格冻结**

```python
class FutureMambaPytorch(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.base = PI0Pytorch(config)
        self.futuremamba = FutureMambaPluginPytorch(config)
        self.freeze_base()

    def train(self, mode: bool = True):
        super().train(mode)
        self.base.eval()
        self.futuremamba.train(mode)
        return self
```

`freeze_base()` 设置所有 base 参数 `requires_grad_(False)`、`eval()`，记录初始 checksum。所有 base forward 位于 `torch.no_grad()`，返回插件的 hidden/KV 显式 `detach()`；不得使用 `torch.inference_mode()`。

插件按以下顺序构造记忆输入：最后有效 VLM token → `vlm_memory_in_proj`；上一周期实际执行 action 的 masked mean 与 last-valid 拼接 → `executed_action_encoder`；二者拼接 → `memory_input_fusion` → `MemoryBackend` → `memory_token_proj`。

- [ ] **步骤 5：实现高噪声硬交接**

```python
handoff_steps = math.ceil(float(handoff_ratio) * int(num_steps))
dt = torch.tensor(-1.0 / num_steps, device=noise.device)
x_t = noise
for step in range(num_steps):
    time = torch.full((noise.shape[0],), 1.0 + step * float(dt), device=noise.device)
    if step < handoff_steps:
        velocity = self.progress_velocity(frozen_prefix, memory_token, x_t, time)
    else:
        velocity = self.base.denoise_step(state, frozen_prefix.pad_mask, frozen_prefix.kv_cache, x_t, time)
    x_t = x_t + dt * velocity
```

主路径不得求和两个速度场，不得在 `step >= handoff_steps` 计算 Progress Expert，不得让 Action Expert 读取 Memory Token。

- [ ] **步骤 6：运行模型测试**

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=src \
  .venv-futuremamba/bin/pytest -q src/openpi/models_pytorch/futuremamba_test.py
```

预期：全部通过；冻结 checksum 未变；只有 `futuremamba.*` 获得梯度。

- [ ] **步骤 7：提交复合模型**

```bash
git add src/openpi/models_pytorch/futuremamba.py src/openpi/models_pytorch/futuremamba_test.py
git commit -m "feat(FutureMamba): 实现 PyTorch 记忆与硬交接"
```

---

### 任务 8：迁移完整 episode 损失与因果 BPTT

**文件：**
- 修改：`src/openpi/models_pytorch/futuremamba.py`
- 修改：`src/openpi/models_pytorch/futuremamba_test.py`
- 修改：`src/openpi/training/episode_data_loader.py`
- 创建：`src/openpi/training/episode_data_loader_pytorch_test.py`

- [ ] **步骤 1：编写失败测试，锁定 PyTorch EpisodeBatch**

```python
def test_episode_batch_to_torch_preserves_query_masks(batch):
    out = episode_batch_to_torch(batch, torch.device("cpu"))
    assert out.query_mask.dtype == torch.bool
    assert out.action_mask.dtype == torch.bool
    assert out.executed_action_mask.dtype == torch.bool
    assert out.actions.dtype == torch.float32
    assert out.observation.state.shape[:2] == out.query_mask.shape
```

- [ ] **步骤 2：编写失败测试，锁定损失公式与归一化**

```python
def test_loss_is_mean_per_episode_then_mean_per_batch(model, uneven_batch):
    losses = model.compute_episode_loss(uneven_batch, noise=fixed_noise, time=fixed_time)
    expected = torch.stack([manual_episode_mean(0), manual_episode_mean(1)]).mean()
    torch.testing.assert_close(losses["loss"], expected)


def test_boundary_target_is_stop_gradient(model, episode_batch):
    result = model.compute_episode_loss(episode_batch)
    result["loss"].backward()
    assert all(p.grad is None for p in model.base.parameters())
    assert any(p.grad is not None for p in model.futuremamba.parameters())
```

- [ ] **步骤 3：运行测试验证失败**

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=src \
  .venv-futuremamba/bin/pytest -q \
  src/openpi/training/episode_data_loader_pytorch_test.py \
  src/openpi/models_pytorch/futuremamba_test.py -k 'loss or episode or padding'
```

预期：FAIL；缺少 PyTorch batch 与 episode loss。

- [ ] **步骤 4：实现 episode 数据转换与有效长度 memory scan**

`episode_batch_to_torch()` 递归把 observation、actions、action mask、executed actions、query/reset mask 移到目标设备。拒绝 `Q_b=0` 和非右侧 padding。MemoryBackend 对每条 episode 只处理 `x[b,:Q_b]`；不得以零输入推进 padding query。跨 query 不 detach，保留完整 BPTT。

- [ ] **步骤 5：实现三个损失项**

`compute_episode_loss()` 保持现有语义：

```python
flow_loss = masked_episode_mean((progress_velocity - target_velocity).square().mean(dim=-1))
handoff_loss = config.handoff_loss_weight * masked_episode_mean(
    (boundary_state - target_boundary).square().mean(dim=-1)
)
boundary_loss = config.boundary_loss_weight * masked_episode_mean(
    (progress_boundary - action_boundary.detach()).square().mean(dim=-1)
)
loss = flow_loss + handoff_loss + boundary_loss
```

Flow Matching time 只采样高噪声区间 $[1-K/N,1]$；handoff 使用真实前 $K$ 步 Progress rollout 得到 $x_K$；padding action/query 均不进入分母。

- [ ] **步骤 6：运行 episode loss 测试**

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=src \
  .venv-futuremamba/bin/pytest -q \
  src/openpi/training/episode_data_loader_pytorch_test.py \
  src/openpi/models_pytorch/futuremamba_test.py
```

预期：全部通过；2-query 梯度能从第二个 query 回到第一个 query 的 memory input。

- [ ] **步骤 7：提交 episode 训练图**

```bash
git add src/openpi/models_pytorch/futuremamba.py \
  src/openpi/models_pytorch/futuremamba_test.py \
  src/openpi/training/episode_data_loader.py \
  src/openpi/training/episode_data_loader_pytorch_test.py
git commit -m "feat(训练图): 迁移完整 episode 因果损失"
```

---

### 任务 9：实现确定性 frozen conditioning cache

**文件：**
- 创建：`src/openpi/training/futuremamba_conditioning_cache.py`
- 创建：`src/openpi/training/futuremamba_conditioning_cache_test.py`
- 创建：`scripts/precompute_futuremamba_prefix.py`

- [ ] **步骤 1：编写失败测试，锁定 cache manifest**

```python
def test_cache_rejects_base_checksum_mismatch(tmp_path):
    write_cache(tmp_path, manifest=manifest(base_checkpoint_checksum="a"), tensors=tensors)
    with pytest.raises(ValueError, match="base_checkpoint_checksum"):
        read_cache(tmp_path, expected=manifest(base_checkpoint_checksum="b"))


def test_cache_requires_complete_prefix_kv_for_boundary_loss(tmp_path):
    incomplete = tensors_with_layers((0, 6, 11, 17))
    with pytest.raises(ValueError, match="complete prefix KV"):
        write_cache(tmp_path, manifest=manifest(), tensors=incomplete)


def test_random_augmentation_disables_cache():
    with pytest.raises(ValueError, match="deterministic preprocessing"):
        validate_cache_mode(train_image_augmentation=True)
```

- [ ] **步骤 2：运行测试验证失败**

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=src \
  .venv-futuremamba/bin/pytest -q src/openpi/training/futuremamba_conditioning_cache_test.py
```

预期：FAIL；cache 模块不存在。

- [ ] **步骤 3：实现分片 cache 与严格 manifest**

每个 episode shard 保存 `last_valid_hidden`、`prefix_mask`、Action Expert 全部逐层 K/V、episode ID 与 query ID。manifest 必须包含 base weights、assets、tokenizer/config、预处理、层映射、dtype 的 checksum。读取时逐字段严格比较；任何不匹配立即拒绝。

- [ ] **步骤 4：实现预计算 CLI 并运行 fake cache smoke**

```bash
PYTHONPATH=src .venv-futuremamba/bin/python scripts/precompute_futuremamba_prefix.py \
  --config futuremamba_pi05_libero_mem_mamba2 \
  --base-checkpoint /home/ubuntu/lgd/CoRL2026/checkpoints/pi05_libero_libero_pi05_2000_pytorch \
  --output-dir /tmp/futuremamba-prefix-cache-smoke \
  --max-episodes 1 --dtype float32
```

预期：生成 manifest 和至少一个 episode shard；重新读取得到同 shape/dtype/checksum。

- [ ] **步骤 5：运行 cache 测试并提交**

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=src \
  .venv-futuremamba/bin/pytest -q src/openpi/training/futuremamba_conditioning_cache_test.py
git add src/openpi/training/futuremamba_conditioning_cache.py \
  src/openpi/training/futuremamba_conditioning_cache_test.py scripts/precompute_futuremamba_prefix.py
git commit -m "feat(训练缓存): 添加冻结 prefix 条件缓存"
```

---

### 任务 10：实现插件 checkpoint 与 PyTorch 训练入口

**文件：**
- 创建：`src/openpi/training/futuremamba_checkpoint.py`
- 创建：`src/openpi/training/futuremamba_checkpoint_test.py`
- 创建：`scripts/train_futuremamba_pytorch.py`
- 创建：`scripts/train_futuremamba_pytorch_test.py`
- 修改：`src/openpi/training/config.py`

- [ ] **步骤 1：编写失败测试，锁定 checkpoint 内容与严格恢复**

```python
def test_checkpoint_contains_plugin_only(tmp_path, model, optimizer):
    save_futuremamba_checkpoint(tmp_path, model, optimizer, scheduler, step=2, metadata=metadata)
    assert {p.name for p in tmp_path.iterdir()} == {
        "plugin.safetensors", "optimizer.pt", "scheduler.pt", "rng_state.pt", "metadata.json"
    }
    assert all(name.startswith("futuremamba.") for name in safe_open_keys(tmp_path / "plugin.safetensors"))


@pytest.mark.parametrize("field", ["memory_backend", "memory_state_schema_version", "base_checkpoint_checksum"])
def test_restore_rejects_identity_mismatch(tmp_path, field):
    bad = dict(metadata)
    bad[field] = "wrong"
    with pytest.raises(ValueError, match=field):
        load_futuremamba_checkpoint(tmp_path, model, optimizer, scheduler, expected_metadata=bad)
```

- [ ] **步骤 2：运行测试验证失败**

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=src \
  .venv-futuremamba/bin/pytest -q src/openpi/training/futuremamba_checkpoint_test.py
```

预期：FAIL；checkpoint 模块不存在。

- [ ] **步骤 3：实现原子保存与严格恢复**

保存前断言 base checksum 等于初始化 checksum；SafeTensors key 必须全部以 `futuremamba.` 开头。`metadata.json` 必须显式包含 `schema_version`、`base_checkpoint_uri`、`base_checkpoint_checksum`、`base_assets_checksum`、`mamba_repo_commit`、`memory_backend`、`memory_state_schema_version`、`memory_config`、`progress_depth`、`progress_layer_mapping`、`handoff_ratio`、`num_denoise_steps`、`executed_horizon`、`loss_weights`、`training_dtype`、`state_dtypes`、`kernel_mode`、`torch_version`、`triton_version`、`cuda_version`、`gpu_name` 和 `compute_capability`。恢复时先逐字段校验 metadata，再以 `strict=True` 加载插件 key；禁止 `strict=False`，禁止从 Mamba-2 恢复 Mamba-3。

- [ ] **步骤 4：注册 Stage B 配置并实现训练循环**

注册 `futuremamba_pi05_libero_mem_mamba2`：base 指向转换目录，完整 episode loader，float32，`handoff_ratio=0.2`、`num_denoise_steps=10`、`executed_horizon=5`。训练入口：

```python
trainable = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
if not trainable or any(not name.startswith("futuremamba.") for name, _ in trainable):
    raise RuntimeError(f"invalid trainable parameters: {[name for name, _ in trainable]}")
optimizer = torch.optim.AdamW([p for _, p in trainable], ...)
```

每步调用 `compute_episode_loss()`，只裁剪插件梯度；checkpoint 恢复同步 optimizer、scheduler、Torch CPU/CUDA RNG 与 data iterator step。

- [ ] **步骤 5：编写并运行 2-step → restore → 4-step smoke**

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=src \
  .venv-futuremamba/bin/pytest -q scripts/train_futuremamba_pytorch_test.py
```

测试使用 tiny fake base/backend，验证第 2 step 保存后恢复到第 4 step：plugin 参数变化，base checksum 不变，optimizer step 连续，固定 seed 的 loss 序列与不中断运行一致。

- [ ] **步骤 6：运行真实小 batch float32 smoke**

```bash
PYTHONPATH=src .venv-futuremamba/bin/python scripts/train_futuremamba_pytorch.py \
  futuremamba_pi05_libero_mem_mamba2 \
  --num-train-steps 2 --batch-size 1 --pytorch-training-precision float32 \
  --wandb-enabled false --overwrite
```

预期：2 步前向、反向和保存成功；日志列出 trainable 仅 `futuremamba.*`；base checksum 前后相同；loss 和梯度均为有限值。

- [ ] **步骤 7：提交训练与 checkpoint**

```bash
git add src/openpi/training/config.py \
  src/openpi/training/futuremamba_checkpoint.py \
  src/openpi/training/futuremamba_checkpoint_test.py \
  scripts/train_futuremamba_pytorch.py scripts/train_futuremamba_pytorch_test.py
git commit -m "feat(训练): 添加纯 PyTorch FutureMamba 训练恢复"
```

---

## 阶段 D：在线 Policy 与闭环迁移

### 任务 11：迁移 FutureMambaPolicy 到显式 PyTorch state

**文件：**
- 修改：`src/openpi/policies/futuremamba_policy.py`
- 修改：`src/openpi/policies/futuremamba_policy_test.py`

- [ ] **步骤 1：重写失败测试，锁定在线状态结构**

```python
def test_snapshot_is_detached_clone_with_identity(policy):
    policy.infer(observation_with_executed_prefix)
    snapshot = policy.snapshot_state()
    assert snapshot.backend_id == "mamba2"
    assert snapshot.state_schema_version == policy.model.memory.state_schema_version
    assert snapshot.query_count == 1
    mutate_policy_state(policy)
    assert_state_equal(snapshot.memory, snapshot_clone)


def test_restore_rejects_backend_or_batch_mismatch(policy):
    snapshot = dataclasses.replace(policy.snapshot_state(), backend_id="mamba3_siso")
    with pytest.raises(ValueError, match="backend"):
        policy.restore_state(snapshot)


def test_reset_clears_memory_and_executed_history(policy):
    policy.infer(observation_with_executed_prefix)
    policy.reset()
    snapshot = policy.snapshot_state()
    assert snapshot.query_count == 0
    assert snapshot.executed_action_mask.count_nonzero() == 0
    assert_state_is_zero(snapshot.memory)
```

- [ ] **步骤 2：运行测试验证旧 JAX state 不满足合同**

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=src \
  .venv-futuremamba/bin/pytest -q src/openpi/policies/futuremamba_policy_test.py
```

预期：FAIL；现有实现返回 JAX state，缺少 backend/schema/executed history。

- [ ] **步骤 3：实现 FutureMambaPolicyState 与 PyTorch 推理**

```python
@dataclasses.dataclass(frozen=True)
class FutureMambaPolicyState:
    backend_id: str
    state_schema_version: int
    memory: MemorySnapshot
    executed_actions: torch.Tensor
    executed_action_mask: torch.BoolTensor
    episode_count: int
    query_count: int
    client_id: str
```

`infer()` 完成 transforms → PyTorch device → `sample_actions_with_memory()` → 保存新 memory 与真实 executed-action history → output transforms。`snapshot_state()` 递归 detach/clone 到 CPU；`restore_state()` 严格校验 backend/schema/layers/shape/dtype/batch。`fork()` 共享只读模型权重但创建独立 state 和 client ID。

- [ ] **步骤 4：运行 Policy 测试**

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=src \
  .venv-futuremamba/bin/pytest -q src/openpi/policies/futuremamba_policy_test.py \
  src/openpi/policies/libero_policy_test.py
```

预期：全部通过；只反馈实际执行 prefix，不反馈 action chunk tail。

- [ ] **步骤 5：提交 Policy 迁移**

```bash
git add src/openpi/policies/futuremamba_policy.py \
  src/openpi/policies/futuremamba_policy_test.py src/openpi/policies/libero_policy_test.py
git commit -m "feat(策略): 迁移 FutureMamba 在线 PyTorch 状态"
```

---

### 任务 12：严格加载插件 bundle 并验证 WebSocket 隔离

**文件：**
- 修改：`src/openpi/policies/policy_config.py`
- 创建：`src/openpi/policies/policy_config_futuremamba_test.py`
- 修改：`src/openpi/serving/websocket_policy_server_test.py`

- [ ] **步骤 1：编写失败测试，锁定 bundle 检测**

```python
def test_plugin_bundle_is_not_misdetected_as_plain_pi0(tmp_path):
    write_bundle(tmp_path, base_uri=base_dir, backend="mamba2")
    policy = create_trained_policy(config, tmp_path, pytorch_device="cpu")
    assert isinstance(policy, FutureMambaPolicy)


def test_loader_rejects_base_checksum_mismatch(tmp_path):
    write_bundle(tmp_path, base_checksum="bad")
    with pytest.raises(ValueError, match="base_checkpoint_checksum"):
        create_trained_policy(config, tmp_path, pytorch_device="cpu")
```

- [ ] **步骤 2：运行测试验证失败**

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=src \
  .venv-futuremamba/bin/pytest -q src/openpi/policies/policy_config_futuremamba_test.py
```

预期：FAIL；当前 loader 只识别 `model.safetensors`。

- [ ] **步骤 3：实现显式 bundle 加载**

若 checkpoint 根目录同时含 `plugin.safetensors` 和 `metadata.json`，先读 metadata，解析 `base_checkpoint_uri`，核对 base weights/assets checksum；构造 `FutureMambaPytorch`，严格加载 base `model.safetensors` 和 plugin。若只有 `model.safetensors`，保持普通 PI0 路径。不得依据宽泛类名或含糊 backend 自动猜测。

- [ ] **步骤 4：扩展真实连接隔离测试**

启动 `WebsocketPolicyServer`，创建两个 fork：A infer → feedback → infer，B 只 infer；断言 A/B query count 和 memory checksum 独立。发送 reset 后，服务端只有在 memory 与 executed history 全清零后返回 `{"reset": True}`。未知 backend/schema 返回结构化错误。

- [ ] **步骤 5：运行 loader 与 WebSocket 测试**

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=src \
  .venv-futuremamba/bin/pytest -q \
  src/openpi/policies/policy_config_futuremamba_test.py \
  src/openpi/serving/websocket_policy_server_test.py \
  packages/openpi-client/src/openpi_client/websocket_client_policy_test.py
```

预期：全部通过。

- [ ] **步骤 6：提交 bundle 与服务迁移**

```bash
git add src/openpi/policies/policy_config.py \
  src/openpi/policies/policy_config_futuremamba_test.py \
  src/openpi/serving/websocket_policy_server_test.py
git commit -m "feat(部署): 严格加载 FutureMamba 插件 bundle"
```

---

### 任务 13：迁移 LIBERO runner、历史因果评测与 profile

**文件：**
- 修改：`examples/libero_mem/main.py`
- 修改：`examples/libero_mem/eval_history_pairs.py`
- 修改：`examples/libero_mem/history_pairs_test.py`
- 修改：`examples/libero_mem/run_experiment_matrix.py`
- 修改：`examples/libero_mem/run_experiment_matrix_test.py`
- 修改：`scripts/profile_futuremamba.py`

- [ ] **步骤 1：编写失败测试，拒绝旧 backend 名称**

```python
def test_matrix_uses_versioned_memory_backends():
    rows = build_matrix()
    backends = {row["config"]["memory_backend"] for row in rows if "memory_backend" in row["config"]}
    assert "mamba" not in backends
    assert "mamba2" in backends
    assert "mamba3_siso" in backends


def test_rollout_log_records_state_schema_and_runtime():
    row = build_episode_log(..., policy_metadata=metadata)
    assert row["memory_backend"] == "mamba2"
    assert row["memory_state_schema_version"] >= 1
    assert row["mamba_repo_commit"] == "77069de5cdb55cbe98b670889c80df211e031039"
```

- [ ] **步骤 2：运行测试验证失败**

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=src \
  .venv-futuremamba/bin/pytest -q \
  examples/libero_mem/run_experiment_matrix_test.py \
  examples/libero_mem/history_pairs_test.py
```

预期：FAIL；旧矩阵仍含 `mamba` 或旧 state schema。

- [ ] **步骤 3：迁移 runner 与因果 state**

runner 保持 reset → infer → 执行前 `executed_horizon` 动作 → feedback → infer。history pair evaluator 保存/恢复新的 `FutureMambaPolicyState`；correct、reset、truncated、shuffled、swapped 五种条件使用相同当前 observation 与 noise。

- [ ] **步骤 4：迁移实验矩阵与 PyTorch profile**

主表使用 Mamba-2；后端代际表加入独立初始化的 Mamba-3 SISO。硬交接消融含 `K=0`、中间 K、`K=N`、全程 memory conditioning 和软融合。将 `scripts/profile_futuremamba.py` 从 JAX loader 改为任务 12 的 PyTorch bundle loader，并增加 `--output`；JSON 固定输出 trainable params、每层/总 state bytes、memory step median/p95、action chunk median/p95、训练/推理峰值显存、kernel mode、GPU/Torch/Triton/CUDA/Mamba commit。缺失的实测项必须报错，不能写假值或沿用 JAX fake loader。

- [ ] **步骤 5：运行评测工具回归**

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=src \
  .venv-futuremamba/bin/pytest -q \
  examples/libero_mem/run_experiment_matrix_test.py \
  examples/libero_mem/history_pairs_test.py \
  examples/libero_mem/metrics_test.py
```

预期：全部通过，配置和结果中不存在 `memory_backend="mamba"`。

- [ ] **步骤 6：提交评测迁移**

```bash
git add examples/libero_mem/main.py examples/libero_mem/eval_history_pairs.py \
  examples/libero_mem/history_pairs_test.py examples/libero_mem/run_experiment_matrix.py \
  examples/libero_mem/run_experiment_matrix_test.py scripts/profile_futuremamba.py
git commit -m "feat(评测): 迁移 FutureMamba 后端与状态元数据"
```

---

## 阶段 E：Mamba-3 硬门与干净切换

### 任务 14：实现 Mamba-3 SISO wrapper 与 RTX 5090 十项硬门

**文件：**
- 修改：`src/openpi/models_pytorch/mamba_memory.py`
- 修改：`src/openpi/models_pytorch/mamba_memory_test.py`
- 创建：`scripts/check_mamba3_rtx5090.py`

- [ ] **步骤 1：编写失败测试，锁定 Mamba-3 原生 state**

```python
def test_mamba3_state_schema_and_dtypes(cuda_device):
    model = Mamba3SisoMemoryBackend(config).to(cuda_device)
    state = model.initial_state(2, device=cuda_device, dtype=torch.bfloat16)
    angle, ssm, key, value = state.layers[0]
    assert angle.dtype == torch.float32
    assert ssm.dtype == torch.float32
    assert key.dtype == torch.bfloat16
    assert value.dtype == torch.bfloat16
    assert key.shape[1] == 1


def test_mamba2_checkpoint_cannot_restore_as_mamba3(mamba2_snapshot, mamba3):
    with pytest.raises(ValueError, match="backend"):
        mamba3.restore(mamba2_snapshot, batch_size=1, device=torch.device("cuda"))
```

- [ ] **步骤 2：运行测试验证失败**

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=src \
  .venv-futuremamba/bin/pytest -q src/openpi/models_pytorch/mamba_memory_test.py -k mamba3
```

预期：FAIL；Mamba-3 wrapper 不存在。

- [ ] **步骤 3：实现官方 Mamba-3 SISO wrapper**

直接实例化官方 `Mamba3(is_mimo=False, mimo_rank=1, chunk_size=64)`。sequence 调官方 `forward`；step 调官方 `step(x, angle, ssm, key, value)`，其中 `x` 为 `[B,D]`，不得错误 unsqueeze。Block 外层保持与 Mamba-2 相同的 Add → RMSNorm → mixer 与 final RMSNorm；state 参数树和 schema 独立。

- [ ] **步骤 4：实现十项硬门脚本**

`check_mamba3_rtx5090.py` 依次运行并记录：依赖、设备 256-step、10 seeds 前向、10 seeds 反向与 optimizer update、sequence/step/state parity、因果、partial/full reset、长度 2/2048 固定 state bytes、OpenPI 2→4 step checkpoint、LIBERO-Mem 部署。parity 门限固定 normalized RMSE `<=1e-3` 且 cosine `>=0.9999`。

输出 `mamba3_gate.json`：

```json
{
  "backend": "mamba3_siso",
  "status": "passed",
  "mamba_repo_commit": "77069de5cdb55cbe98b670889c80df211e031039",
  "checks": {"dependency": true, "device": true, "forward": true, "backward": true, "parity": true, "causality": true, "reset": true, "fixed_state": true, "openpi": true, "deployment": true}
}
```

任何门失败时 `status` 必须为 `unsupported_on_current_stack`、退出非 0，并保留失败异常、版本、GPU 与命令；不得自动改记为 Mamba-2。

- [ ] **步骤 5：运行硬门**

```bash
PYTHONPATH=src .venv-futuremamba/bin/python scripts/check_mamba3_rtx5090.py \
  --output /tmp/futuremamba-mamba3-gate.json \
  --config=futuremamba_pi05_libero_mem_mamba3_siso \
  --libero-task-id=0 --libero-trials=1
```

预期分支：

- 全部 10 门通过：允许注册和训练 `mamba3_siso`；
- 任一门失败：Mamba-3 标记 `unsupported_on_current_stack`，正式实验继续 Mamba-2，任务本身以“失败被准确门控”验收，不修改官方 kernel。

- [ ] **步骤 6：提交 Mamba-3 门控**

```bash
git add src/openpi/models_pytorch/mamba_memory.py \
  src/openpi/models_pytorch/mamba_memory_test.py scripts/check_mamba3_rtx5090.py
git commit -m "feat(记忆后端): 添加 Mamba-3 SISO 硬件门控"
```

---

### 任务 15：完成 Mamba-2 真实集成 smoke

**文件：**
- 不新增生产代码
- 验证：训练、Policy、WebSocket、LIBERO-Mem、历史因果与 profile

- [ ] **步骤 1：运行定向 PyTorch 回归**

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=src \
  .venv-futuremamba/bin/pytest -q \
  src/openpi/models_pytorch/futuremamba_config_test.py \
  src/openpi/models_pytorch/pi0_pytorch_test.py \
  src/openpi/models_pytorch/mamba_memory_test.py -k 'not mamba3' \
  src/openpi/models_pytorch/progress_expert_test.py \
  src/openpi/models_pytorch/futuremamba_test.py \
  src/openpi/training/episode_data_loader_pytorch_test.py \
  src/openpi/training/futuremamba_conditioning_cache_test.py \
  src/openpi/training/futuremamba_checkpoint_test.py \
  scripts/train_futuremamba_pytorch_test.py \
  src/openpi/policies/futuremamba_policy_test.py \
  src/openpi/serving/websocket_policy_server_test.py
```

预期：全部选中测试通过，0 failed；记录 exact pass count。

- [ ] **步骤 2：启动真实 Policy server**

```bash
PYTHONPATH=src .venv-futuremamba/bin/python scripts/serve_policy.py policy:checkpoint \
  --policy.config=futuremamba_pi05_libero_mem_mamba2 \
  --policy.dir=checkpoints/futuremamba_pi05_libero_mem_mamba2/2 \
  --port=8000
```

预期：服务加载 base + plugin，metadata 显示 `memory_backend=mamba2`，health endpoint 为 200。

- [ ] **步骤 3：运行 WebSocket 生命周期 smoke**

从客户端执行 reset → infer → executed-action feedback → infer。预期：第二次 query state checksum 改变；reset 后恢复零状态；两个连接 state 不串线。

- [ ] **步骤 4：运行单个 LIBERO-Mem episode**

```bash
PYTHONPATH=. .venv-futuremamba/bin/python examples/libero_mem/main.py \
  --host 127.0.0.1 --port 8000 --task-suite-name libero_mem \
  --task-ids 0 --num-trials-per-task 1 --replan-steps 5 \
  --results-path /tmp/futuremamba-mamba2-rollout.jsonl
```

预期：完成真实 reset/infer/feedback 闭环，无 state/schema/timeout 错误；JSONL 含 backend、schema、Mamba commit 和 handoff diagnostics。

- [ ] **步骤 5：生成固定历史对并运行因果 smoke**

```bash
PYTHONPATH=. .venv-futuremamba/bin/python examples/libero_mem/build_history_pairs.py \
  --repo-id futuremamba/libero_mem_long_val \
  --max-pairs-per-task 1 \
  --output /tmp/futuremamba-history-pairs.jsonl \
  --similarity-threshold 0.95 --noise-seed 0
PYTHONPATH=. .venv-futuremamba/bin/python examples/libero_mem/eval_history_pairs.py \
  --pairs /tmp/futuremamba-history-pairs.jsonl \
  --results /tmp/futuremamba-mamba2-history.jsonl \
  --policy-path checkpoints/futuremamba_pi05_libero_mem_mamba2/2 \
  --policy-config futuremamba_pi05_libero_mem_mamba2 \
  --task-suite-name libero_mem \
  --truncated-k 2 --shuffle-seed 0
```

预期：manifest 每个 task 至多 1 对；correct/reset/truncated/shuffled/swapped 条件均运行，当前 observation/noise checksum 相同，state checksum 符合干预定义。

- [ ] **步骤 6：运行 profile smoke**

```bash
PYTHONPATH=src .venv-futuremamba/bin/python scripts/profile_futuremamba.py \
  --config=futuremamba_pi05_libero_mem_mamba2 \
  --checkpoint-root=checkpoints/futuremamba_pi05_libero_mem_mamba2/2 \
  --warmup-queries=10 --measured-queries=100 \
  --output=/tmp/futuremamba-mamba2-profile.json
```

预期：所有规定指标为实测值；kernel mode 明确为 fallback 或 fused，不用伪造值填充失败项。

---

### 任务 16：干净切换生产路径并删除 JAX 专用实现

**文件：**
- 删除：本计划“最终删除”清单中的 JAX FutureMamba 文件
- 修改：`src/openpi/training/config.py`
- 修改：所有旧 import、配置、测试与实验矩阵调用方
- 修改：`pyproject.toml`、`uv.lock`

- [ ] **步骤 1：在隔离门已通过的前提下更新主依赖**

主项目固定 Torch `2.9.1`、Triton `3.5.1` 与官方 Mamba commit；为 Mamba build 提供匹配 runtime Torch 的 extra build dependency。运行：

```bash
uv lock
uv sync --frozen
```

预期：主锁文件解析成功，`python environments/futuremamba/check_runtime.py` 的 Mamba-2 项全部通过。Mamba-3 import 结果仍由硬门决定。

- [ ] **步骤 2：删除旧 JAX FutureMamba 实现**

```bash
git rm \
  src/openpi/models/mamba.py src/openpi/models/mamba_test.py \
  src/openpi/models/progress_expert.py src/openpi/models/progress_expert_test.py \
  src/openpi/models/futuremamba.py src/openpi/models/futuremamba_test.py \
  src/openpi/models/futuremamba_config.py \
  scripts/train_futuremamba.py scripts/train_futuremamba_test.py
```

- [ ] **步骤 3：迁移所有调用方并验证无旧标识**

```bash
python - <<'PY'
from pathlib import Path
roots = [Path("src"), Path("scripts"), Path("examples")]
needles = ["openpi.models.futuremamba", "openpi.models.mamba", 'memory_backend="mamba"', '"memory_backend": "mamba"']
hits = []
for root in roots:
    for path in root.rglob("*.py"):
        text = path.read_text()
        for needle in needles:
            if needle in text:
                hits.append(f"{path}: {needle}")
if hits:
    raise SystemExit("\n".join(hits))
PY
```

预期：退出码 0，无旧 import、别名或含糊 backend。普通 OpenPI JAX `Pi0` 保留，不删除上游通用 JAX 支持。

- [ ] **步骤 4：运行主环境回归**

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=src .venv/bin/pytest -q \
  src/openpi/models/pi0_test.py \
  src/openpi/models_pytorch \
  src/openpi/policies \
  src/openpi/serving \
  src/openpi/training/episode_data_loader_test.py \
  src/openpi/training/episode_data_loader_pytorch_test.py \
  scripts/train_futuremamba_pytorch_test.py \
  examples/libero_mem
```

预期：全部选中测试通过，0 failed；普通 JAX PI0 和普通 PyTorch PI0 未回归。

- [ ] **步骤 5：提交干净切换**

```bash
git add -A
git commit -m "refactor(FutureMamba)!: 切换为唯一纯 PyTorch 实现" \
  -m "删除 FutureMamba 专用 JAX Mamba、Progress Expert 与训练入口。" \
  -m "BREAKING CHANGE: memory_backend=mamba 已移除，请使用 mamba2 或 mamba3_siso。"
```

---

### 任务 17：最终验收与论文证据边界

**文件：**
- 不修改生产代码
- 输出：命令日志、转换 parity JSON、Mamba-3 gate JSON、rollout JSONL、history JSONL、profile JSON

- [ ] **步骤 1：运行格式和静态检查**

```bash
.venv/bin/ruff check \
  src/openpi/models_pytorch src/openpi/policies src/openpi/serving \
  src/openpi/training scripts examples/libero_mem
.venv/bin/ruff format --check \
  src/openpi/models_pytorch src/openpi/policies src/openpi/serving \
  src/openpi/training scripts examples/libero_mem
```

预期：退出码 0。

- [ ] **步骤 2：重跑基座转换 parity**

运行任务 2 步骤 6 的命令。预期：四类比较全部满足 MAE/MaxAE 门限。

- [ ] **步骤 3：重跑 Mamba-2 完整验收矩阵**

重复任务 15 的定向测试、2-step 训练、WebSocket、单 episode、历史因果和 profile。预期：全部成功；base checksum 始终不变。

- [ ] **步骤 4：按 gate 结果处理 Mamba-3**

若 `/tmp/futuremamba-mamba3-gate.json` 为 `passed`，用 `mamba3_siso` 重复任务 15 全矩阵；若为 `unsupported_on_current_stack`，验证正式实验配置未选择 Mamba-3，并保留失败证据。不得把 Mamba-2 结果标为 Mamba-3。

- [ ] **步骤 5：核对完成定义**

逐项确认：严格转换、Mamba-2 sequence/step/reset/snapshot/checkpoint、单次 prefix cache、Memory Token 隔离、硬交接、episode BPTT、Policy/WebSocket/runner/matrix/profile、新 backend/schema、旧 JAX 路径删除均有上述命令证据。

论文结果分为四类：FutureMamba 架构收益、Mamba-2/3 后端代际、硬交接收益、正确/破坏记忆因果性。Mamba-3 未通过 RTX 5090 门时，只报告为当前栈不支持，不进入主表或部署结论。

---

## 执行顺序与停止条件

1. 任务 1 是依赖硬门；Mamba-2 import/forward 不可用时，修复环境后才能进入任务 2。
2. 任务 2 的 float32 基座 parity 是 Stage B 硬门；不满足门限时，不训练插件。
3. 任务 4–13 先完成可投稿的 Mamba-2 全链路。
4. 任务 14 的 Mamba-3 任一门失败，不阻塞 Mamba-2 迁移完成，但必须禁止 Mamba-3 正式训练和部署。
5. 只有任务 15 的真实 smoke 通过后，才执行任务 16 的不可逆 JAX 专用路径删除。
6. 每个任务提交前只运行其定向测试；任务 17 统一运行格式、回归和真实 smoke，避免重复消耗 GPU 与数据环境。
