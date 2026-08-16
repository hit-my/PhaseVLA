# FutureMamba 纯 PyTorch 与 RoboMME 仿真设计规格

- **日期：** 2026-08-11
- **状态：** RoboMME 方案已确认；此前 LIBERO 仿真设计废止，纯 PyTorch 核心迁移成果继续复用。
- **目标：** 在 RoboMME 官方任务适配版 $\pi_{0.5}$ 上训练查询级 Mamba 进程记忆与轻量 Progress Expert，并以 RoboMME 官方 ManiSkill/SAPIEN 闭环完成全部仿真训练、评测和论文证据。
- **冻结基座：** RoboMME 官方微调 `pi05_baseline`，严格转换为 PyTorch 后冻结 VLM 与 Action Expert。
- **主后端：** 官方 Mamba-2；官方 Mamba-3 SISO 只有通过 RTX 5090 硬门后才进入独立消融。
- **唯一仿真平台：** RoboMME。LIBERO-Mem、LIBERO-Long 和普通 LIBERO 不再承担训练、验证、测试或论文结论。

## 1. 决策摘要

采用以下路线：

1. **保留 RoboMME 官方仿真闭环。** 数据、任务定义、ManiSkill/SAPIEN 环境、`examples/robomme/eval.py`、成功判定、视频与结果聚合均来自 RoboMME；只替换 WebSocket 后的策略服务。
2. **策略与仿真使用两个软件环境。** PyTorch 策略环境运行冻结 $\pi_{0.5}$、Mamba 和 Progress Expert；RoboMME 环境运行模拟器。两者只通过 msgpack/WebSocket 交换 reset、history buffer、当前观测和 action chunk。
3. **训练图统一为 PyTorch。** $\pi_{0.5}$、Mamba、Progress Expert 和损失位于同一 PyTorch autograd 图；VLM 与 Action Expert 参数冻结。
4. **Mamba 只在策略查询时更新一次。** 主路径输入为当前查询的 VLM 最后有效 token 与上一记忆状态；执行 action chunk 的 16 个底层控制步期间不更新 Mamba，也不把整段 buffer 当作 16 次递推。
5. **Progress Expert 负责高噪声阶段。** 默认从微调 Action Expert 的 Uniform-6 对应层初始化，读取相同层的只读 VLM K/V 与投影后的 Memory Token，完成前 $K$ 个去噪步；冻结 Action Expert 从同一 scheduler 位置接管。
6. **主损失为 Progress flow loss 加终端动作损失。** Action Expert 参数冻结，但终端去噪不得放入 `no_grad()`，以便梯度从最终动作穿过冻结 Action Expert 回传至 $x_K$、Progress Expert 与 Mamba。
7. **先接入官方 Mamba-2，再门控 Mamba-3 SISO。** 两者共享上层 `MemoryBackend` 合同，不共享参数或状态；Mamba-3 未通过 RTX 5090 硬门时不进入主表或部署结论。
8. **干净切换。** RoboMME 路径验收后删除 FutureMamba 专用 JAX 实现与 LIBERO 专用 FutureMamba runner；OpenPI 上游通用 JAX/LIBERO 示例不在删除范围内。
## 2. 已观察到的仓库与环境事实

### 2.1 本仓库状态

本工作区已实现或正在迁移纯 PyTorch FutureMamba 核心：

- `src/openpi/models_pytorch/pi0_pytorch.py`：PyTorch $\pi_{0.5}$、prefix KV、单步速度与 Euler 采样；
- `src/openpi/models_pytorch/mamba_memory.py`：Mamba-2 及消融后端；
- `src/openpi/models_pytorch/progress_expert.py`：只读 prefix 的轻量专家；
- `src/openpi/models_pytorch/futuremamba.py`：冻结基座、查询级记忆与硬交接；
- `src/openpi/policies/futuremamba_policy.py`：显式 PyTorch 状态生命周期；
- `src/openpi/training/futuremamba_checkpoint.py`：仅插件 checkpoint。

旧 `examples/libero_mem/` 只代表此前实验方向，不能再作为 RoboMME 仿真完成证据。后续应扩展现有 PyTorch 路径并新增 RoboMME 适配，不创建第三套模型框架。

### 2.2 RoboMME 固定来源与协议

正式实验固定以下来源：

- 策略仓库：`RoboMME/robomme_policy_learning@ecf086c3be7c2223167d9bb2f6ef1f0a6e24353b`；
- 仿真子模块：`RoboMME/robomme_benchmark@856bc3a189d4172f3f47dbee4424d585f8d78db3`；
- 原始数据：`Yinpei/robomme_data_h5`；
- 官方微调基座：`Yinpei/pi05_baseline`；
- 官方对比 checkpoint：`Yinpei/mme_vla_suite`。

官方事实：16 个任务分为 Counting、Permanence、Reference、Imitation 四组，每组 4 个任务；训练集每任务 100 条示范，验证集和测试集每任务各 50 个 episode。官方 $\pi_{0.5}$ 配置预测 20 步 action chunk，评测客户端默认实际执行前 16 步，最大底层控制步为 1300。

官方 WebSocket 客户端发送三类消息：`{"reset": true}`、带 `add_buffer=true` 的历史帧 buffer、当前观测 infer。原服务响应分别为 `reset_finished`、`add_buffer_finished` 和 `actions`。FutureMamba 服务必须与该客户端兼容，同时允许结构化扩展元数据；不得要求修改 ManiSkill/SAPIEN 环境语义。

### 2.3 RoboMME 基座 checkpoint

唯一正式基座是下载后的 RoboMME `pi05_baseline` 后期 checkpoint，例如：

```text
runs/ckpts/pi05_baseline/pi05_baseline/79999
```

它是 Orbax/JAX checkpoint，包含 `params/` 与 RoboMME `assets/robomme/norm_stats.json`。转换必须同时迁移 `RoboMMEInputs`、`RoboMMEOutputs`、224 × 224 图像变换、8 维 joint-angle/state 语义和归一化统计。原 LIBERO checkpoint 不得作为 RoboMME 正式基座。

### 2.4 依赖风险

当前项目声明：

- Python $\ge 3.11$；
- PyTorch `2.7.1`；
- `uv.lock` 中 Triton `3.3.1`。

官方 Mamba 当前发布线的事实：

- `mamba_ssm` 同一个 Python 包同时提供 `Mamba2` 与 `Mamba3`，同一环境不能并装两个标签；
- 两个后端统一固定为官方 `v2.3.2`，提交 `77069de5cdb55cbe98b670889c80df211e031039`，避免后端对比同时混入包版本差异；
- `v2.3.2` 的 `pyproject.toml` 要求 Triton $\ge 3.5.0$，并引入 TileLang、Quack Kernels 和 Apache TVM FFI；
- PyTorch `2.9.1` 官方依赖 Triton `3.5.1`，作为隔离兼容性试验的首选精确版本组合；
- 固定官方 Mamba 版本的仓库许可证为 Apache-2.0；
- 当前主项目声明 PyTorch `2.7.1`、锁定 Triton `3.3.1`，不能作为 Mamba-3 kernel 可用性的证据。
- 当前 shell 默认 `nvcc` 为 CUDA `11.5`，低于 Mamba `setup.py` 要求的 CUDA `11.6`，且不能为 RTX 5090 生成 `sm_120`；机器另有 `/usr/local/cuda-12.8`，隔离构建必须显式设置 `CUDA_HOME=/usr/local/cuda-12.8`，不能依赖默认 `PATH`。

结论：先在隔离环境验证 PyTorch `2.9.1` + Triton `3.5.1` + Mamba `v2.3.2`，源码构建时显式使用 CUDA 12.8 工具链并强制生成官方扩展。Mamba-2 与 Mamba-3 共用该环境和官方提交；隔离门通过后再一次性更新主 `pyproject.toml` 与 `uv.lock`，禁止在两个不同 Torch/Mamba 栈上生成论文对比结果。

## 3. 不变量与范围边界

### 3.1 必须保持的 FutureMamba 语义

每个第 $q$ 次 action-chunk 查询执行：

$$
H_q,\mathcal C_q=\operatorname{VLM}_{\mathrm{frozen}}(o_q,r_q,\ell),
$$

其中 $H_q$ 是最后一层 prefix hidden states，$\mathcal C_q$ 是逐层 prefix KV cache。主方法只使用最后一个有效 prefix token：

$$
z_q=H_q[p_q],
\qquad
p_q=\max\{j:\operatorname{prefix\_mask}_{q,j}=1\},
$$

$$
e_q=W_z z_q,
\qquad
(h_q,S_q)=\operatorname{MemoryStep}(e_q,S_{q-1}),
\qquad
m_q=W_m h_q.
$$

约束：

- 第 $q$ 次 `infer` 恰好触发一次 Mamba 更新；首个查询以前状态为零；
- RoboMME `add_buffer` 只声明从上次查询到当前查询的观测历史与 `exec_start_idx`，不得直接推进 Mamba；
- 官方模型预测 20 步，但环境默认执行前 16 步；未执行的 4 步不得进入任何历史动作基线；
- 主 FutureMamba 不把 past actions 送入 Mamba；`pi0.5 + past actions` 是独立对比基线；
- episode 开始、环境 reset 或远程 `{"reset": true}` 时清零 memory、query count 与临时 buffer；
- padding query 不更新状态、不产生损失；
- Memory Token 只进入 Progress Expert，不写回冻结 VLM，也不进入冻结 Action Expert；
- 状态大小不随 episode 长度增长，不跨 episode 传播。

### 3.2 高噪声硬交接

时间约定保持 OpenPI 现状：$t=1$ 为噪声，$t=0$ 为动作。设总求解步数为 $N$，交接比例为 $\rho$：

$$
K=\lceil\rho N\rceil,
\qquad
\Delta t=-\frac1N.
$$

前 $K$ 步只运行 Progress Expert：

$$
x_{i+1}=x_i+\Delta t\,v_P(x_i,t_i,\mathcal C_q,m_q),
\qquad 0\le i<K.
$$

后 $N-K$ 步只运行冻结 Action Expert：

$$
x_{i+1}=x_i+\Delta t\,v_A(x_i,t_i,\mathcal C_q),
\qquad K\le i<N.
$$

Progress Expert 的作用是产生交接状态 $x_K$。主路径禁止：

- `v = v_A + v_P`；
- 将 `v_P` 拼接或残差注入 Action Expert；
- 在 $i\ge K$ 时继续无效计算 Progress Expert；
- 让 Action Expert 读取 Memory Token。

软融合、残差融合和 `action_memory_full` 只能保留为显式消融配置。

### 3.3 不在本次实现范围内

- 修改 RoboMME 任务定义、ManiSkill/SAPIEN 动力学、成功判定或 train/val/test split；
- 使用 LIBERO 结果替代任何 RoboMME 验收或论文结果；
- 新增未来图像、未来状态或额外人工阶段标签监督；
- 引入跨 episode 终身记忆；
- 修改冻结 $\pi_{0.5}$ Action Expert 架构；
- 首版接入 Mamba-3 MIMO；
- 为迁移方便保留 JAX/PyTorch 混合训练或永久兼容 shim。

## 4. 纯 PyTorch 系统架构

### 4.1 顶层模块

新模型为 `FutureMambaPytorch(nn.Module)`，包含：

```text
FutureMambaPytorch
├── base: PI0Pytorch                         # 冻结
│   ├── PaliGemma / VLM                     # 冻结
│   └── Gemma Action Expert                 # 冻结
└── futuremamba: FutureMambaPluginPytorch    # 可训练
    ├── vlm_memory_in_proj
    ├── memory: MemoryBackend
    ├── memory_token_proj
    └── progress_expert: ProgressExpertPytorch
```

冻结必须同时满足：

1. 基座所有参数 `requires_grad=False`；
2. 基座保持 `eval()`，避免 dropout 或训练态行为；
3. VLM prefix forward 使用 `torch.no_grad()`；所有交给插件的 hidden/KV 在退出 `no_grad` 后显式 `detach`，保持为普通 tensor；终端损失中的冻结 Action Expert 去噪只冻结参数，不能用 `no_grad()` 或 `inference_mode()`，因为梯度必须传回 $x_K$；
4. optimizer 参数列表只包含 `futuremamba.*`；
5. 每次训练启动时断言没有非插件参数可训练；
6. 保存前断言冻结参数 checksum 与加载时一致。

不能只依靠“不把参数放入 optimizer”来伪冻结。

### 4.2 Prefix 编码接口

从 `PI0Pytorch` 提取无行为变化的 helper：

```python
@dataclass
class FrozenPrefix:
    hidden: torch.Tensor       # [B, C, D_vlm]
    pad_mask: torch.BoolTensor # [B, C]
    kv_cache: object           # HF cache，逐层 [B, H_kv, C, D_head]

@torch.no_grad()
def encode_frozen_prefix(observation) -> FrozenPrefix: ...
```

同一次 prefix forward 的结果同时供以下路径使用：

- `hidden` 与 `pad_mask`：抽取记忆输入的最后有效 token；
- 原始 `kv_cache`：冻结 Action Expert；
- 映射后的只读 cache view：Progress Expert。

`FrozenPrefix` 返回前必须递归 `detach` hidden 与 KV tensor。若 cache 实现会复用可变存储，还必须为 Progress Expert 创建独立的 detached clone；`detach` 只切断基座梯度，不能用 `inference_mode` tensor 代替普通只读 conditioning tensor，因为插件反向需要保存这些值来计算可训练 Q/K/V 和投影参数的梯度。

必须避免：

- 为 Progress Expert 再运行一次 VLM；
- 重新训练一套 context K/V projection；
- 让 Progress Expert 修改 Action Expert 使用的原始 cache；
- 用 padding 后的固定末位代替最后有效 token。

### 4.3 PyTorch Progress Expert

`ProgressExpertPytorch` 使用 Action Expert 相同的 decoder block 类型，但只保留 Uniform-6 层：

- hidden width、head 数、KV head 数、head dim、MLP dim、激活函数和 AdaRMS 时间条件与 Action Expert 相同；
- 主设置 `progress_depth=6`，层索引由 Action Expert 深度上均匀覆盖首尾得到；
- 每个 Progress layer 读取同索引 VLM 层的只读 K/V；
- Progress Expert 的可对应参数从 RoboMME 微调 Action Expert 对应层初始化；输入/输出投影、Memory Token 投影等无对应参数使用项目标准初始化；
- 随机初始化整个 Progress Expert 只作为显式消融，不作为主设置。

对第 $j$ 个 Progress layer，从冻结 prefix cache 读取第 $r(j)$ 层的 K/V，并构造该 Progress layer 自己的 cache 索引。输入 token 顺序为：

```text
[prefix cache] [single Memory Token] [action tokens]
```

Action tokens 通过该 Progress layer 自己的 Q/K/V 投影；Memory Token 通过该层独立的 K/V 投影，不产生 query；冻结 prefix K/V 只读。标准因果顺序保证 action query 可以读取 Memory Token 和当前上下文。Progress Expert 的输出只取最后 `action_horizon` 个 action token，再通过独立 `action_out_proj` 得到速度。

实现必须使用独立的 cache view 或 detached clone。若 Hugging Face cache API 会原地追加 token，则不得把 Action Expert 的原始 cache 对象直接传入 Progress Expert。

### 4.4 RoboMME 连续窗口训练

RoboMME 官方预处理样本含 `epis_idx`、`step_idx`、`exec_start_idx`，动作目标为 20 步 chunk。FutureMamba 数据加载器必须先按 `(epis_idx, step_idx)` 重建 episode 内顺序，再抽取连续 query 窗口：

1. 每个窗口只来自一个 `epis_idx`，`step_idx` 严格递增；乱序、重复或跨 episode 拼接立即报错；
2. 每个 episode 的初始 Mamba 状态为零；窗口若从 episode 中间开始，必须先无梯度 burn-in 到窗口起点，或加载由同一 checkpoint 生成且通过 checksum 校验的起始状态；不得假装中间窗口从零开始；
3. 窗口内按 query 顺序运行 Mamba，长度 $W$ 为可配置 truncated BPTT 窗口；窗口结束后 detach state，禁止把梯度跨窗口或跨 episode 传播；
4. `query_mask` 只允许右侧 padding，padding query 不推进状态、不产生损失；
5. 冻结 prefix hidden 与选定层 K/V 为普通 detached tensor；
6. 每个有效 query 以 20 步 action target 训练，但在线闭环只执行前 16 步；
7. 先按每个窗口的有效 query 取均值，再对 batch 取均值。

冻结 prefix KV 可采用两条数值等价路径：

- **主训练路径：** 以 RoboMME 官方确定性预处理为输入，按 episode/query 预计算最后有效 token、prefix mask 与完整逐层 KV；
- **在线回退路径：** 逐 query 运行冻结 prefix，完成该 query 损失后立即释放 cache。

缓存 manifest 必须记录 RoboMME 策略仓库 commit、仿真子模块 commit、原始数据 checksum、基座权重与 assets checksum、tokenizer、图像预处理、层映射、dtype、episode/query ID。任一身份不匹配即拒绝读取。

## 5. 统一 MemoryBackend 合同

### 5.1 公共接口

Mamba-2 与 Mamba-3 只共享上层语义，不共享参数树或底层 state shape：

```python
class MemoryBackend(nn.Module):
    backend_id: str
    state_schema_version: int

    def initial_state(self, batch_size, *, device, dtype): ...

    def forward_sequence(
        self,
        x: torch.Tensor,      # [B, Q, D]
        *,
        query_mask: torch.BoolTensor | None = None,  # 仅允许右侧 padding
    ) -> tuple[torch.Tensor, object]: ...

    @torch.no_grad()
    def step(
        self,
        x: torch.Tensor,      # [B, D]
        state: object,
    ) -> tuple[torch.Tensor, object]: ...
```

合同要求：

- `forward_sequence` 只用于从零状态开始的完整 episode；返回 padding 后的 `[B,Q,D]` 与每条 episode 的最后有效状态；
- `query_mask=None` 表示全部 query 有效；否则每行必须是连续有效前缀，非右侧 padding 立即报错；
- padding query 永远不调用官方 mixer，因而不改变 convolution、SSM、angle、K 或 V state；
- `step` 返回 `[B,D]` 与下一状态；
- 对同一权重、有效输入和零初始状态，整段与逐步输出在已声明 tolerance 内一致；
- state 是显式、可 clone、可移设备、可序列化的嵌套 tensor，不把官方可变 `InferenceParams` 字典暴露给 Policy；
- `reset(mask)` 只清零指定 batch 行；
- snapshot 必须 detach、clone 并带上 backend/schema 元数据；
- restore 必须严格校验 backend、schema、层数、shape、dtype 和 batch size。

### 5.2 公共超参数

Mamba-2 与 Mamba-3 的初始对比使用相同的高层容量配置：

```text
d_model = 1024
depth = 2
d_state = 128
expand = 2
headdim = 64
ngroups = 1
rms_norm = true
residual_in_fp32 = true
fused_add_norm = false
```

Mamba-2 额外参数：

```text
d_conv = 4
use_mem_eff_path = false   # 正确性阶段；fused 路径需另做 parity
```

Mamba-3 首版额外参数：

```text
is_mimo = false
mimo_rank = 1
rope_fraction = 0.5
chunk_size = 64
is_outproj_norm = false
```

Mamba-2 和 Mamba-3 参数量天然不同。论文必须报告实际参数量、FLOPs、state bytes、延迟和显存；若需要等参数对比，另设 width-matched 消融，不能宣称默认配置已等参数。

### 5.3 官方 Block 语义

每层使用官方 `Block` 的 Add → Norm → Mixer 语义，`d_intermediate=0`；多层之后应用官方 final RMSNorm。允许在本项目 wrapper 中显式组织 residual、final norm 和外置 state，但禁止复制或改写 Mamba-2/3 的状态更新方程与 CUDA/Triton/CuTe kernel。

训练路径调用官方 mixer 的 sequence `forward`。在线路径调用官方 mixer 的 `step`，wrapper 只负责：

- 每层输入的 residual 与 norm；
- 将每层原生 state 放入统一外层容器；
- 统一输入/输出的 `[B,D]` 形状；
- final RMSNorm；
- reset、snapshot 与 restore。

## 6. Mamba-2 阶段

### 6.1 固定来源

- 仓库：`https://github.com/state-spaces/mamba`
- 标签：`v2.3.2`
- 提交：`77069de5cdb55cbe98b670889c80df211e031039`
- 类：`mamba_ssm.modules.mamba2.Mamba2`
- 许可证：Apache-2.0

Mamba-2 与 Mamba-3 必须来自同一安装包、同一提交和同一 Torch/Triton 环境。阶段顺序仍是先完成 Mamba-2 全链路，再开启 Mamba-3 硬门；统一来源只消除依赖混杂，不允许跳过 Mamba-2 验收。

### 6.2 原生状态

每层 Mamba-2 state：

```text
conv_state: [B, d_ssm + 2 × ngroups × d_state, d_conv]
ssm_state:  [B, nheads, headdim, d_state]
```

官方 `step` 接收 `[B,1,D]`，返回 `[B,1,D]` 并原地更新 state。项目 wrapper 对外暴露 `[B,D]`，在无梯度推理路径中负责 unsqueeze/squeeze。snapshot 必须 clone，不能把仍会原地变化的 tensor 引用交给调用方。

### 6.3 正确性优先路径

首个闭环版本：

- sequence 训练使用 `use_mem_eff_path=False`；
- 在线 step 在可选 CUDA op 缺失时允许走官方文件内的 PyTorch fallback；
- 不因追求 fused kernel 而改变数值合同；
- `causal-conv1d`、`selective_state_update` 或 fused sequence path 只有通过 fallback parity 后才启用。

Mamba-2 是完整可投稿的主后端，不是一次性 scaffold。若 Mamba-3 门控失败，正式实验继续使用 Mamba-2，并如实把 Mamba-3 记为未满足当前硬件部署条件。

## 7. Mamba-3 阶段

### 7.1 固定来源

- 仓库：`https://github.com/state-spaces/mamba`
- 标签：`v2.3.2`
- 提交：`77069de5cdb55cbe98b670889c80df211e031039`
- 许可证：Apache-2.0
- 模式：SISO，`is_mimo=False`

Mamba-3 不通过 JAX 重写，不复制论文公式自行实现。正式版本必须直接调用该固定官方源码中的 `Mamba3`、SISO sequence kernel 和 `step` 路径。

### 7.2 原生状态

每层 Mamba-3 SISO state：

```text
angle_dt_state: [B, nheads, num_rope_angles]          float32
ssm_state:      [B, nheads, headdim, d_state]         float32
k_state:        [B, 1, nheads, d_state]               model dtype
v_state:        [B, nheads, headdim]                  model dtype
```

Mamba-3 state 与 Mamba-2 state 不兼容。所谓“保持同一接口切换”只指 `MemoryBackend` 的输入/输出语义一致，不允许：

- 把 Mamba-2 state 恢复为 Mamba-3 state；
- 用 `strict=False` 静默加载 Mamba-2 memory 权重；
- 为复用 checkpoint 而增加无论文意义的参数映射；
- 把 Mamba-2 训练结果标成 Mamba-3。

Mamba-3 实验从同一冻结 $\pi_{0.5}$ 基座和相同随机种子协议重新初始化、重新训练插件。Mamba-2 checkpoint 可用于验证上层接口，但不能作为 Mamba-3 memory 初始化。

### 7.3 RTX 5090 硬门

官方 `Mamba3.step` 注释明确说明目前仅在 H100 上测试。RTX 5090 必须依次通过：

1. **依赖门：** 隔离环境中固定版本可安装，`Mamba3`、SISO sequence kernel、rotary step 和 CuTe step kernel 均可导入；不得出现静默 `None` 后仍继续训练。
2. **设备门：** 在 Compute Capability 12.0 上成功分配全部 state，并完成至少 256 个连续 step；无 illegal instruction、kernel launch failure 或 silent fallback。
3. **前向门：** 对 10 个固定随机种子，sequence forward 输出 shape 正确且无 NaN/Inf。
4. **反向门：** 对 10 个固定随机种子，输入与所有可训练参数梯度存在且无 NaN/Inf；完成 optimizer step 后参数确实变化。
5. **一致性门：** float32 下 sequence 输出与从零状态逐 token step 输出满足 normalized RMSE $\le10^{-3}$ 且 cosine similarity $\ge0.9999$；最后 state 的对应值也满足同一 normalized RMSE 门限。
6. **因果门：** 修改第 $q+1$ 个及之后的输入不改变前 $q$ 个输出。
7. **reset 门：** 部分 batch reset 只影响被 reset 的行；完全 reset 后输出与新建零状态一致。
8. **固定大小门：** 长度 2 与长度 2048 的最终 state tree、shape 和总字节数完全相同。
9. **OpenPI 集成门：** 完成 2-step fake episode 训练、保存、恢复到第 4 step，并证明只更新 `futuremamba.*`。
10. **部署门：** 使用固定 RoboMME 客户端完成至少一个真实 episode 的 `reset → add_buffer → infer → 执行 16 步 → add_buffer → infer`；无状态串线、协议错误或超时。

任何一门失败时：

- Mamba-3 后端状态标记为 `unsupported_on_current_stack`；
- 不自动回退成 Mamba-2 却仍记录 `backend=mamba3`；
- 不修改官方 kernel 源码伪造支持；
- 保留失败日志、软件版本、GPU 与复现命令；
- 正式实验继续使用 Mamba-2。

## 8. 基座权重迁移

### 8.1 严格转换

使用 OpenPI 官方 `examples/convert_jax_model_to_pytorch.py` 将任务适配的 Orbax checkpoint 转成 PyTorch `safetensors`。转换前扫描 JAX 参数树：

- 若存在 `lora_a` / `lora_b`，标准转换器可能静默丢弃 adapter；必须先做可验证的 LoRA merge 或使用 LoRA-aware 转换器；
- 若不存在 LoRA，仍需检查全部基座参数 key、shape 和 dtype；
- `assets/`、normalization stats、tokenizer/config 标识必须随转换结果复制并记录 checksum；
- PyTorch 训练精度先固定为 `float32` 完成 parity，再决定插件训练是否用 `bfloat16`。

### 8.2 转换验收

固定同一批预处理 observation、action、noise 和 flow time，分别运行 JAX 与 PyTorch 基座：

- prefix 最后有效 token；
- 选定层的 prefix K/V；
- Action Expert 单步速度 $v_t$；
- 固定噪声的完整 10-step action chunk。

float32 初始门限：

```text
mean absolute error <= 1e-4
max absolute error  <= 5e-4
```

同时报告速度场 cosine similarity 与 action chunk MAE。数值门通过后，使用相同 RoboMME test episode 和 seed 比较 JAX/PyTorch 成功率；PyTorch 版不得低于 JAX 版 95% 置信区间下界。未通过时不得开始 FutureMamba 正式训练。

## 9. 训练目标与精度

对每个有效查询，在高噪声区间采样 $\tau$：

$$
x_\tau=(1-\tau)A_{\mathrm{gt}}+\tau\epsilon,
\qquad
v^*=\epsilon-A_{\mathrm{gt}}.
$$

Progress Expert 的主监督为：

$$
\mathcal L_{\mathrm{flow}}^P
=\left\|v_P(x_\tau,h_q,\tau)-v^*\right\|_2^2.
$$

从同一初始噪声运行前 $K$ 个 Progress step 得到 $x_K$，再让冻结 Action Expert 完成剩余 $N-K$ 步得到 $\hat A$。总损失为：

$$
\mathcal L
=\mathcal L_{\mathrm{flow}}^P
+\lambda_{\mathrm{term}}\left\|\hat A-A_{\mathrm{gt}}\right\|_2^2.
$$

约束：

- Action Expert 参数 `requires_grad=False`，但从 $x_K$ 到 $\hat A$ 的计算图必须保留；
- `terminal_loss_weight=0` 是 flow-only 消融，主设置必须显式记录非零权重；
- padding query 和 padding action 不参与分母；
- 若显存不足，可对 Action Expert 剩余步使用 gradient checkpointing，或在预注册的 batch 子集计算 terminal loss；不得用 `no_grad()` 静默切断主实验梯度；
- 旧 handoff/boundary velocity imitation 可保留为额外消融，但不替代上述主目标。

精度策略：基座转换与 scan/step parity 使用 float32；Mamba-2 首个训练 smoke 使用 float32；bfloat16 只有在 loss、梯度和固定输入输出稳定后启用；官方要求 float32 的 state 保持 float32；checkpoint metadata 记录训练 dtype、state dtype、kernel 和 fallback。

## 10. Checkpoint 与 Policy 状态

### 10.1 训练 checkpoint

FutureMamba 训练 checkpoint 至少包含：

```text
plugin.safetensors
optimizer.pt
scheduler.pt
rng_state.pt
metadata.json
```

`metadata.json` 必须包含：

```text
schema_version
base_checkpoint_uri
base_checkpoint_checksum
base_assets_checksum
robomme_policy_commit
robomme_benchmark_commit
robomme_dataset_checksum
robomme_task_suite
mamba_repo_commit
memory_backend              # mamba2 / mamba3_siso / gru / ...
memory_state_schema_version
memory_config
progress_depth              # 主设置 6
progress_layer_mapping     # Uniform-6
handoff_ratio
num_denoise_steps
prediction_horizon          # RoboMME 主设置 20
execution_horizon           # RoboMME 官方评测主设置 16
loss_weights
training_dtype
state_dtypes
kernel_mode                 # fallback / triton / cute
torch_version
triton_version
cuda_version
gpu_name
compute_capability
```

加载时严格拒绝 backend、schema、shape、RoboMME 身份或 base checksum 不匹配。禁止用 `strict=False` 吞掉 memory 参数差异。

训练 checkpoint 默认只存插件与基座引用，避免重复保存冻结大模型。部署 bundle 可以打包基座与插件，但仍保留独立 checksum。RoboMME `assets/robomme/norm_stats.json` 的 checksum 必须与转换 manifest 和服务 metadata 一致。

### 10.2 在线 Policy state

Policy snapshot 独立于训练 checkpoint，包含：

- backend ID 与 state schema；
- 每层 memory state 的 detached clone；
- 当前 episode/query 计数；
- 当前连接的 `client_id`；
- 最近一次观测、history buffer 和 action chunk 的 checksum 及执行边界，仅用于诊断，不作为主 Mamba 输入。

RoboMME 的 `add_buffer` 只维护协议所需历史帧；它不能隐式推进 Mamba。每个 WebSocket 连接必须 fork 独立 Policy 实例。收到 `{"reset": true}` 后，只有在 memory、query count 和临时 history buffer 均清零时才返回 `{"reset_finished": true}`。

## 11. 配置与命名

弃用含糊的：

```text
memory_backend = "mamba"
```

改为：

```text
memory_backend = "mamba2"
memory_backend = "mamba3_siso"
```

Mamba-3 MIMO 未来若接入，使用单独标识 `mamba3_mimo`，不能由 `mamba3_siso` 配置隐式开启。

所有实验记录与 checkpoint 使用相同命名。论文主表必须把 Mamba 版本写到方法配置或脚注中，不能把 Mamba-2 结果统称为 Mamba-3。

## 12. 目标文件与干净切换

### 12.1 新增 PyTorch 模块

建议新增：

```text
src/openpi/models_pytorch/futuremamba_config.py
src/openpi/models_pytorch/mamba_memory.py
src/openpi/models_pytorch/progress_expert.py
src/openpi/models_pytorch/futuremamba.py
scripts/train_futuremamba_pytorch.py
```

测试对应放在同目录或项目既有测试位置。最终命名在实现计划中按仓库惯例确定，但禁止同时保留两个可被生产配置选中的 FutureMamba 类。

### 12.2 修改现有路径

- `src/openpi/models_pytorch/pi0_pytorch.py`：提取 prefix 与单步 Action Expert helper；RoboMME $\pi_{0.5}$ 数值路径保持等价；
- `src/openpi/models_pytorch/gemma_pytorch.py`：提供只读 cache view 与 Uniform-6 Progress Expert 所需的稳定接口；
- `src/openpi/training/config.py`：注册 RoboMME FutureMamba PyTorch config、20 步 prediction horizon 与 16 步 execution horizon；
- `src/openpi/training/episode_data_loader.py`：读取 RoboMME `epis_idx`、`step_idx`、`exec_start_idx` 并构造连续窗口；
- `src/openpi/policies/policy_config.py`：加载 PyTorch FutureMamba bundle 和 RoboMME assets；
- `src/openpi/policies/futuremamba_policy.py`、`src/openpi/serving/websocket_policy_server.py`：兼容 RoboMME `reset`、`add_buffer`、infer 协议并保持连接级 state 隔离；
- 创建 `examples/robomme/` 适配与测试：只负责策略服务器启动、RoboMME payload 转换、metadata 与 rollout 诊断，不复制环境或成功判定；
- 创建 `scripts/run_robomme_experiment_matrix.py`、`scripts/profile_futuremamba.py` 的 RoboMME 入口；
- `pyproject.toml` / `uv.lock`：只在隔离依赖门通过后更新。

### 12.3 删除旧路径

PyTorch 功能、回归、闭环和 checkpoint 迁移全部通过后：

- 删除 FutureMamba 专用的自研 JAX `SelectiveMamba`；
- 删除 JAX `FutureMamba` 与 JAX `ProgressExpert`；
- 删除只服务旧实现的配置、权重过滤器和测试；
- 将 `scripts/train_futuremamba.py` 切换或重命名为唯一 PyTorch 入口；
- 更新实验矩阵，不再接受 `memory_backend="mamba"`；
- 不保留兼容别名、双写 checkpoint 或运行时自动迁移。

## 13. 验收矩阵

### 13.1 单元与数值合同

必须覆盖：

- 最后有效 token 抽取、空 mask 与查询级单次更新；
- `add_buffer` 不更新 Mamba，连续 16 个底层动作不更新 Mamba；
- Progress layer mapping、prefix padding、Memory Token 隔离；
- 原始 prefix cache 不被 Progress Expert 修改；
- Mamba-2 sequence/step parity、reset、因果性、固定 state bytes；
- Mamba-3 的 10 项硬门；
- $K=0$ 与原 PyTorch $\pi_{0.5}$ 固定噪声数值等价；
- $K=N$ 只调用 Progress Expert；
- 中间 $K$ 的调用次数严格为 `progress=K`、`action=N-K`；
- 冻结参数 checksum 不变；
- 2-step 训练、保存、恢复到第 4 step；
- snapshot/restore 与多连接 state 隔离。

### 13.2 RoboMME 端到端 smoke

Mamba-2 必须先完成：

1. RoboMME 官方 `scripts/dataset_replay.py` 数据回放；
2. 官方 `pi05_baseline` JAX 策略服务与一个 test episode；
3. 同一 checkpoint 转换后的 PyTorch $\pi_{0.5}$ 单回合，成功率、动作和日志与 JAX 基线对齐；
4. FutureMamba PyTorch 服务启动，metadata 包含 RoboMME 两个 commit、`memory_backend=mamba2`、`prediction_horizon=20`、`execution_horizon=16`；
5. RoboMME 客户端 `reset → add_buffer → infer → 执行 16 步 → add_buffer → infer`；
6. Counting Suite 中 `PickXtimes` 与 `BinFill` 各 10 个 validation episode；
7. 结果保存视频、动作 chunk、policy query、memory bytes、handoff step、成功状态和异常日志。

Mamba-3 通过硬门后，使用同一 RoboMME 协议和 seed 独立训练、独立 checkpoint、独立 smoke；不能把 Mamba-2 checkpoint 伪装成 Mamba-3。

### 13.3 性能报告

对 Mamba-2 fallback、Mamba-2 fused（若启用）和 Mamba-3 SISO 分别报告：

- 可训练参数量；
- 每层和总 state bytes；
- 单次 memory step 的 median / p95 latency；
- 完整 action chunk 的 median / p95 latency；
- 峰值训练显存与推理显存；
- kernel 路径与是否发生 fallback；
- GPU、Torch、Triton、CUDA 和官方 Mamba commit。

性能差异不改变正确性门。若 Mamba-3 正确但不满足控制时限，只能作为离线消融，不能作为部署主后端。

## 14. 论文实验解释边界

迁移完成后，论文结果按 RoboMME 官方四类任务报告：

1. **FutureMamba 架构收益：** `mamba2` FutureMamba 对冻结无记忆 RoboMME `pi05_baseline`、`pi0.5 + past actions`、GRU/LSTM/frame-stack 等基线；
2. **官方 memory baseline 对比：** 与 MemER、FrameSamp + Modul、FrameSamp + Expert，以及可获得的 RMT/TTT 变体比较；
3. **任务类别结果：** Counting（BinFill、PickXtimes、SwingXtimes、StopCube）、Permanence、Reference、Imitation 四组及 Overall；
4. **硬交接收益：** 中间 $0<K<N$ 对 $K=0$、$K=N$、无 memory、memory shuffle 和软融合；
5. **记忆因果性：** 保持当前 RoboMME observation 与 noise 不变，交换不同 query 的 memory snapshot，报告早期 velocity、动作误差和最终成功率；
6. **工程代价：** 训练参数、总参数、query/action-chunk 延迟、峰值显存和额外 FLOPs。

不能把 Mamba-3 参数或 kernel 差异当成 FutureMamba 架构贡献，也不能在 Mamba-3 未通过 RTX 5090 门控时宣称已完成官方 Mamba-3 部署。

## 15. 已知风险与处理

| 风险 | 证据 | 处理 |
|---|---|---|
| RoboMME 官方客户端协议与当前 OpenPI 控制消息不同 | 官方使用 `reset` / `add_buffer`，当前服务使用 `__openpi_control__` | 服务端显式兼容两套消息并做协议测试；RoboMME 评测只走官方消息 |
| 20 步预测与 16 步执行被混淆 | 官方数据构建 horizon 为 20，评测 `obs_horizon` 为 16 | 配置分离 `prediction_horizon` / `execution_horizon`；日志同时记录 |
| 随机单帧采样破坏 Mamba 因果性 | 官方 pickle 样本含 episode/step 标识，但默认 Dataset 按样本随机取值 | 新建 episode 索引与连续窗口 sampler；跨 episode 立即拒绝 |
| terminal loss 被 `no_grad()` 静默切断 | Action Expert 参数冻结与输入梯度是两个不同合同 | 冻结参数但保留从 $x_K$ 开始的 autograd；测试 $x_K$ 与插件梯度 |
| OpenPI Torch/Triton 与官方 Mamba 依赖冲突 | 主项目与 Mamba v2.3.2 的 Triton 要求不同 | 使用显式 CUDA 12.8 的隔离 PyTorch 环境；版本写入 metadata |
| Mamba-3 step 未验证 RTX 5090 | 官方源码写明仅在 H100 测试 | 十项硬门；失败则正式实验只用 Mamba-2 |
| JAX checkpoint 转换漂移 | RoboMME 基座是 Orbax，转换还涉及专用 transforms/assets | float32 逐层 parity、速度场 cosine、action MAE、同批 episode 回归 |
| Prefix cache 被两个专家共享时原地修改 | HF cache 可能在 forward 中 update | 独立只读 cache view / detached clone 与 mutation 测试 |
| 训练窗口从 episode 中间错误置零 | truncated BPTT 需要正确起始 state | episode 起点 burn-in 或严格身份的状态缓存；禁止无依据置零 |
| Mamba-2/3 checkpoint 被误混用 | state 与参数结构不同 | backend/schema/checksum 严格加载，禁止 `strict=False` |

## 16. 完成定义

本迁移只有在以下条件全部成立时才算完成：

- RoboMME `pi05_baseline` 已严格转换为 PyTorch，速度场、action chunk、同批 test episode 成功率和延迟均有对照证据；
- 官方 Mamba-2 完整支持 sequence 训练、在线 query step、reset、snapshot/restore 和 plugin checkpoint；
- Progress Expert 使用 RoboMME 基座同一次冻结 prefix 的只读逐层 KV，主设置为 Uniform-6，并保持 Memory Token 隔离；
- 高噪声硬交接、Progress flow loss 与 terminal action loss 在 PyTorch 中端到端可训练；
- RoboMME 官方 WebSocket `reset`、`add_buffer`、infer、16 步执行闭环通过，两个软件环境可独立启动；
- Counting Suite 10-episode validation smoke 通过，并完成 RoboMME 全部 16 任务的 val/test 评测协议；
- 所有结果按 Counting、Permanence、Reference、Imitation 和 Overall 聚合，报告均值、标准差与 95% 置信区间；
- Mamba-3 只有在 RTX 5090 十项硬门全部通过后才标记可用；
- FutureMamba 专用 JAX 实现、LIBERO 专用 FutureMamba runner 与含糊 backend 名称已删除；
- 定向测试、训练 smoke、真实 RoboMME 闭环 smoke、基座 parity、profile 和论文证据均有可复现日志；
- checkpoint metadata 足以区分基座、RoboMME 版本、Mamba 版本、kernel、dtype 和状态 schema。

若 Mamba-3 门控失败，但上述 Mamba-2 RoboMME 路径和干净切换完成，则纯 PyTorch FutureMamba 迁移仍完成；Mamba-3 必须明确记录为当前硬件栈不支持。

## 17. 参考来源

- RoboMME 策略与评测仓库（固定提交）：https://github.com/RoboMME/robomme_policy_learning/tree/ecf086c3be7c2223167d9bb2f6ef1f0a6e24353b
- RoboMME 仿真基准（固定提交）：https://github.com/RoboMME/robomme_benchmark/tree/856bc3a189d4172f3f47dbee4424d585f8d78db3
- RoboMME 论文：https://arxiv.org/abs/2603.04639
- RoboMME 原始数据：https://huggingface.co/datasets/Yinpei/robomme_data_h5
- RoboMME `pi05_baseline`：https://huggingface.co/Yinpei/pi05_baseline
- RoboMME 官方 memory checkpoints：https://huggingface.co/Yinpei/mme_vla_suite
- OpenPI PyTorch 模型与训练：仓库内 `src/openpi/models_pytorch/`、`scripts/train_pytorch.py`。
- 官方 Mamba v2.3.2：https://github.com/state-spaces/mamba/tree/v2.3.2
- 官方 Mamba-3 模块：https://github.com/state-spaces/mamba/blob/v2.3.2/mamba_ssm/modules/mamba3.py
- OpenPI JAX → PyTorch 转换问题 #810：https://github.com/Physical-Intelligence/openpi/issues/810
- OpenPI LoRA 转换问题 #958：https://github.com/Physical-Intelligence/openpi/issues/958
- FutureMamba 方法设计：`docs/superpowers/specs/2026-08-08-futuremamba-design.md`。
