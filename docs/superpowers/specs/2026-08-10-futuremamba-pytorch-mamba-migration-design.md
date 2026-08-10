# FutureMamba 纯 PyTorch 与官方 Mamba-2/3 迁移设计规格

- **日期：** 2026-08-10
- **状态：** 设计已确认，待书面审查；尚未开始迁移实现。
- **目标：** 将现有 FutureMamba 从自研 JAX/Flax Mamba-1 实现迁移到统一的 PyTorch 训练与推理图；先用官方 Mamba-2 打通端到端闭环，再在同一上层接口下门控切换官方 Mamba-3 SISO。
- **冻结基座：** 任务适配后的 OpenPI $\pi_{0.5}$。
- **主硬件：** NVIDIA GeForce RTX 5090，Compute Capability 12.0，32 GB 显存。
- **方法边界：** 保留已确认的查询级进程记忆、Progress Expert 和高噪声硬交接；本次迁移不改变论文研究问题、训练标签或 LIBERO-Mem 评测协议。

## 1. 决策摘要

采用以下路线：

1. **训练图统一为 PyTorch。** $\pi_{0.5}$、记忆模块、Progress Expert、损失和 episode 训练循环全部位于同一 PyTorch autograd 图中。禁止用 JAX 训练基座、再通过跨框架桥接训练 Mamba。
2. **先接入官方 Mamba-2。** 使用 `state-spaces/mamba` 官方模块和官方 `Block` 语义，完成完整 episode 训练、在线单步状态、reset、snapshot/restore、checkpoint 和闭环 rollout。
3. **再接入官方 Mamba-3 SISO。** 保持 FutureMamba 的 `MemoryBackend` 上层合同不变，但 Mamba-2 与 Mamba-3 使用各自原生参数和状态结构，分别从同一冻结基座独立训练。
4. **Progress Expert 继续硬交接。** 它直接负责前 $K$ 个高噪声 flow step，冻结 Action Expert 从 $x_K$ 接管；默认不将两个速度场相加，也不把 $v_P$ 作为残差输入 Action Expert。
5. **Mamba-3 受硬件门控。** 官方 `Mamba3.step` 源码明确标注只在 H100 上测试。RTX 5090 上未通过依赖、kernel、前向/反向和 scan/step parity 之前，不把 Mamba-3 用于正式训练、论文主表或部署。
6. **干净切换，不维护双实现。** PyTorch 版本完成验收后，删除 FutureMamba 专用的自研 JAX Mamba/Progress Expert 路径并迁移全部调用方；OpenPI 上游自身的通用 JAX 支持不在删除范围内。

## 2. 已观察到的仓库与环境事实

### 2.1 现有代码

当前仓库已经包含完整的 JAX/Flax FutureMamba 原型：

- `src/openpi/models/mamba.py`：自研 Mamba-1 风格 selective SSM；
- `src/openpi/models/futuremamba.py`：JAX FutureMamba 与记忆后端；
- `src/openpi/models/progress_expert.py`：JAX Progress Expert；
- `scripts/train_futuremamba.py`：JAX episode 训练入口；
- `src/openpi/policies/futuremamba_policy.py`：有状态策略生命周期；
- LIBERO-Mem 转换、runner、历史因果评测与实验矩阵。

OpenPI 同时已有可训练的 PyTorch $\pi_{0.5}$：

- `src/openpi/models_pytorch/pi0_pytorch.py` 已实现 PyTorch flow-matching 训练、prefix KV cache、`denoise_step` 与 Euler 采样；
- `src/openpi/models_pytorch/gemma_pytorch.py` 已实现 PaliGemma 与 Action Expert 的联合训练路径，以及基于 prefix cache 的 Action Expert 推理路径；
- `scripts/train_pytorch.py` 已提供单机、多卡 DDP/FSDP、保存和恢复训练的基础设施；
- `examples/convert_jax_model_to_pytorch.py` 已提供官方 JAX → PyTorch 基座权重转换入口。

因此迁移应扩展现有 PyTorch 路径，不应新建第三套模型框架。

### 2.2 基座 checkpoint

当前可见的任务适配 checkpoint：

```text
/home/ubuntu/lgd/CoRL2026/checkpoints/pi05_libero_libero_pi05_2000
```

它是 Orbax/JAX checkpoint，包含 `params/` 与 `assets/`，不是可由 PyTorch 直接加载的 `safetensors`。迁移前必须完成一次严格转换和数值验收。

### 2.3 依赖风险

当前项目声明：

- Python $\ge 3.11$；
- PyTorch `2.7.1`；
- `uv.lock` 中 Triton `3.3.1`。

官方 Mamba 当前发布线的事实：

- Mamba-2 基准接入固定为 `v2.2.4`，提交 `95d8aba8a8c75aedcaa6143713b11e745e7cd0d9`；
- Mamba-3 候选接入固定为 `v2.3.2`，提交 `77069de5cdb55cbe98b670889c80df211e031039`；
- `v2.3.2` 的 `pyproject.toml` 要求 Triton $\ge 3.5.0$，并引入 TileLang、Quack Kernels 和 Apache TVM FFI；
- 两个固定版本的仓库许可证均为 Apache-2.0；
- 当前 `/home/ubuntu/lgd/CoRL2026/openpi/.venv` 未安装 `torch`，所以它不能作为 kernel 可用性的证据。

结论：不得直接修改主 `uv.lock` 来“试装” Mamba-3。先在隔离环境完成兼容性试验，再决定主环境升级矩阵。

## 3. 不变量与范围边界

### 3.1 必须保持的 FutureMamba 语义

每个第 $q$ 次 action-chunk 查询执行：

$$
H_q,\mathcal C_q=\operatorname{VLM}_{\mathrm{frozen}}(o_q,r_q,\ell),
$$

其中 $H_q$ 是最后一层 prefix hidden states，$\mathcal C_q$ 是逐层 prefix KV cache。记忆输入使用最后一个有效 prefix token 和上一周期实际执行动作：

$$
z_q=H_q[p_q],
\qquad
p_q=\max\{j:\operatorname{prefix\_mask}_{q,j}=1\},
$$

$$
e_q=W_e\left[W_z z_q\Vert E_a(A_{q-1}^{\mathrm{exec}})\right].
$$

记忆后端更新固定大小状态：

$$
(h_q,S_q)=\operatorname{MemoryStep}(e_q,S_{q-1}),
\qquad
m_q=W_m h_q.
$$

约束：

- 只输入真正下发执行的 action prefix，不输入上一动作块未执行的 tail；
- episode 开始、环境 reset 或显式远程 reset 时清零状态；
- padding query 不更新状态、不产生损失；
- Memory Token 只进入 Progress Expert，不写回冻结 VLM，也不进入冻结 Action Expert；
- 状态大小不随 episode 长度增长。

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

### 3.3 不在本次迁移范围内

- 改变 LIBERO-Mem / LIBERO-Long 数据定义或指标；
- 新增未来图像、未来状态或阶段标签监督；
- 引入跨 episode 终身记忆；
- 修改 $\pi_{0.5}$ Action Expert 架构；
- 首版接入 Mamba-3 MIMO；
- 为迁移方便而保留 JAX/PyTorch 混合训练或永久兼容 shim。

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
    ├── executed_action_encoder
    ├── memory_input_fusion
    ├── memory: MemoryBackend
    ├── memory_token_proj
    └── progress_expert: ProgressExpertPytorch
```

冻结必须同时满足：

1. 基座所有参数 `requires_grad=False`；
2. 基座保持 `eval()`，避免 dropout 或训练态行为；
3. 基座 forward 使用 `torch.no_grad()`；所有交给可训练插件的 hidden/KV tensor 都在退出 `no_grad` 后显式 `detach`，并保持为普通 tensor；禁止用 `torch.inference_mode()` 产生随后需要被 autograd 保存用于插件反向的 inference tensor；
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

`ProgressExpertPytorch` 使用 Hugging Face Gemma 的 Action Expert 同类 decoder block，但深度更小：

- hidden width、head 数、KV head 数、head dim、MLP dim、激活函数和 AdaRMS 时间条件与 Action Expert 相同；
- 主设置 `progress_depth=4`，即约为 `gemma_300m` Action Expert 深度的 $1/4$；
- 层映射覆盖 Action Expert 首尾层，并在中间等距取样；
- Progress Expert 参数独立、可训练；默认保持现有方法的随机初始化，不在迁移时静默改成 Action Expert 权重复制。

对第 $j$ 个 Progress layer，从冻结 prefix cache 读取第 $r(j)$ 层的 K/V，并构造该 Progress layer 自己的 cache 索引。输入 token 顺序为：

```text
[prefix cache] [single Memory Token] [action tokens]
```

Action tokens 通过该 Progress layer 自己的 Q/K/V 投影；Memory Token 通过该层独立的 K/V 投影，不产生 query；冻结 prefix K/V 只读。标准因果顺序保证 action query 可以读取 Memory Token 和当前上下文。Progress Expert 的输出只取最后 `action_horizon` 个 action token，再通过独立 `action_out_proj` 得到速度。

实现必须使用独立的 cache view 或 detached clone。若 Hugging Face cache API 会原地追加 token，则不得把 Action Expert 的原始 cache 对象直接传入 Progress Expert。

### 4.4 Episode 训练的数据流

完整 episode 训练保留因果顺序和跨 query 梯度：

1. 每个 batch 元素是一条 episode，`query_mask` 只允许形如 `[True, ..., True, False, ..., False]` 的右侧 padding；
2. 基座 prefix 特征与被选中的 prefix KV 均为普通 detached tensor，无基座梯度；
3. 每条 episode 只把前 $Q_b=\sum_q\operatorname{query\_mask}_{b,q}$ 个有效 $e_q$ 送入 Mamba；padding query 不能通过零输入或占位输入推进 convolution/SSM state；
4. `forward_sequence` 按有效长度分桶，或逐 episode 对 `x[b,:Q_b]` 调用官方 causal sequence forward；每条 episode 从零状态开始并保持完整跨 query BPTT；
5. 不同长度的输出只在 Mamba forward 之后 padding 回 `[B,Q,D]`，最终 state 来自各自最后一个有效 query；
6. 按有效 query 计算 Progress Expert 的 flow-matching、交接和边界损失；
7. 先按 episode 的有效 query 数取均值，再对 batch 取均值；$Q_b=0$ 的样本在 data loader 阶段拒绝。

由于冻结 prefix KV 占用较大，提供两种数值等价路径：

- **主训练路径：** Stage B 使用确定性的冻结基座 conditioning cache，按 episode 分片保存最后有效 token、prefix mask 和 Action Expert 的全部逐层 prefix KV；Progress Expert 只从该完整 cache 读取层映射选中的 K/V，边界损失复用完整 cache 调用冻结 Action Expert；
- **在线回退路径：** 逐 query 运行冻结 prefix，完成该 query 损失后立即释放 cache，不在 GPU 上保留整条 episode 的全部 VLM cache。

缓存文件必须记录：基座权重 checksum、tokenizer/config checksum、图像预处理配置、层映射、dtype 和 episode/query ID。任一 checksum 不匹配即拒绝读取。若 Stage B 使用随机图像增强，则不能复用离线 prefix cache；主设置因此采用确定性预处理。

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
- 标签：`v2.2.4`
- 提交：`95d8aba8a8c75aedcaa6143713b11e745e7cd0d9`
- 许可证：Apache-2.0

选择 `v2.2.4` 的原因：它包含官方 Mamba-2，同时未在项目依赖中强制 Triton $\ge3.5.0$，适合作为 OpenPI 当前 Torch 2.7.1 / Triton 3.3.1 的第一兼容性基线。版本仍需实际安装和 GPU 验证，不能仅根据依赖声明判断兼容。

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
10. **部署门：** 完成至少一个 LIBERO-Mem 闭环 episode 的 reset → infer → executed-action feedback 循环；无状态串线或超时。

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

若某层因框架 kernel 差异无法满足，必须定位首个分歧层并给出有证据的新门限；不能只因 LIBERO rollout “看起来能跑”就接受转换。数值门通过后，再以相同 seed 运行基座 LIBERO 闭环，确认 PyTorch 基座能力不低于 JAX 基座置信区间下界。

FutureMamba Stage B 只能建立在通过该门的 PyTorch 基座上。

## 9. 训练目标与精度

训练目标保持现有设计：

$$
\mathcal L
=\frac1Q\sum_q
\left(
\mathcal L_{\mathrm{FM}}^{(q)}
+\lambda_h\mathcal L_{\mathrm{handoff}}^{(q)}
+\lambda_b\mathcal L_{\mathrm{boundary}}^{(q)}
\right).
$$

其中：

- `FM` 只监督高噪声区间；
- `handoff` 监督实际前 $K$ 步 rollout 后的 $x_K$；
- `boundary` 用冻结 Action Expert 的切换点速度作为 stop-gradient 目标；
- 冻结基座不接收任何梯度；
- padding query 和 padding action 均不参与归一化分母。

精度策略：

1. 基座转换 parity 与 MemoryBackend scan/step parity 使用 float32；
2. 首个 Mamba-2 训练 smoke 使用 float32；
3. bfloat16 只有在 loss、梯度和固定输入输出均无异常后启用；
4. Mamba state 中官方规定为 float32 的部分保持 float32，不做全局强制转换；
5. checkpoint metadata 必须记录训练 dtype、state dtype、kernel 路径和 fallback 状态。

## 10. Checkpoint 与 Policy 状态

### 10.1 训练 checkpoint

Stage B checkpoint 至少包含：

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
mamba_repo_commit
memory_backend              # mamba2 / mamba3_siso / gru / ...
memory_state_schema_version
memory_config
progress_depth
progress_layer_mapping
handoff_ratio
num_denoise_steps
executed_horizon
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

加载时严格拒绝 backend、schema、shape 或 base checksum 不匹配。禁止用 `strict=False` 吞掉 memory 参数差异。

训练 checkpoint 默认只存插件与基座引用，避免重复保存冻结大模型。部署 bundle 可以打包基座与插件，但仍保留独立 checksum。

### 10.2 在线 Policy state

Policy snapshot 独立于训练 checkpoint，包含：

- backend ID 与 state schema；
- 每层 memory state 的 detached clone；
- 上一周期实际执行动作及其 mask；
- 当前 episode/query 计数；
- batch/client 标识。

WebSocket server 必须按连接 fork 独立 Policy 实例。reset ack 只有在 memory state 与 executed-action history 均已清零后返回。

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

需要修改：

- `src/openpi/models_pytorch/pi0_pytorch.py`：提取 prefix 与单步 Action Expert helper；原 PI0 数值路径保持等价；
- `src/openpi/models_pytorch/gemma_pytorch.py`：提供只读 cache view 与轻量 Gemma expert 所需的稳定接口；
- `src/openpi/training/config.py`：注册纯 PyTorch FutureMamba config；
- `src/openpi/policies/policy_config.py`：加载 PyTorch FutureMamba；
- Policy、WebSocket、LIBERO runner 和实验矩阵：迁移 backend 名称与 state schema；
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

- 最后有效 token 抽取与空 mask；
- 实际执行 action 的 masked mean + last-valid 摘要；
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

### 13.2 端到端 smoke

Mamba-2 必须先完成：

1. fake episode 两步更新；
2. 小型真实 episode batch 前向、反向和恢复；
3. 单个 LIBERO-Mem episode 的本地 Policy 闭环；
4. WebSocket 客户端 reset → infer → feedback → infer；
5. 固定历史对的正确 state / reset state / swapped state 因果检查。

Mamba-3 通过硬门后重复同一矩阵。不能因 Mamba-2 已通过而跳过 Mamba-3 的集成验证。

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

迁移完成后，论文应区分：

1. **FutureMamba 架构收益：** Mamba-2 FutureMamba 对冻结无记忆 $\pi_{0.5}$、GRU/LSTM/frame-stack 等基线；
2. **后端代际差异：** 在相同数据、Progress Expert、交接和训练预算下，Mamba-2 对 Mamba-3 SISO；
3. **硬交接收益：** 中间 $0<K<N$ 对 $K=0$、$K=N$、全程记忆条件和软融合；
4. **记忆因果性：** 正确 state 对 reset、truncated、shuffled 和 swapped state。

不能把 Mamba-3 的参数或 kernel 差异当成 FutureMamba 架构贡献，也不能在 Mamba-3 未通过 RTX 5090 门控时宣称已完成官方 Mamba-3 部署。

## 15. 已知风险与处理

| 风险 | 证据 | 处理 |
|---|---|---|
| OpenPI Torch/Triton 与 Mamba-3 依赖冲突 | Torch 2.7.1 / Triton 3.3.1；Mamba v2.3.2 要求 Triton $\ge3.5.0$ | 隔离环境验证；通过前不改主锁文件 |
| Mamba-3 step 未验证 RTX 5090 | 官方源码写明仅在 H100 测试 | 10 项硬门；失败则保留 Mamba-2 |
| JAX checkpoint 转换漂移 | 基座为 Orbax；官方 issue 曾报告精度和 LoRA 丢失问题 | float32 逐层 parity、LoRA 扫描、闭环基座回归 |
| Prefix cache 被两个专家共享时被原地修改 | HF cache 可能在 forward 中 update | 独立只读 cache view / detached clone；mutation 测试 |
| 完整 episode 的 prefix KV 占用过大 | 每个 query 含多图像 token 和逐层 KV | 确定性离线 conditioning cache，或逐 query 即用即释放 |
| Mamba-2/3 checkpoint 被误混用 | state 与参数结构不同 | backend/schema/checksum 严格加载，禁止 `strict=False` |
| 迁移同时改变方法导致实验不可比 | PyTorch 重写可能顺手改变初始化、交接或损失 | 保持方法不变量；新选择只能作为显式消融 |

## 16. 完成定义

本迁移只有在以下条件全部成立时才算完成：

- 任务适配 $\pi_{0.5}$ 已严格转换为 PyTorch，数值与闭环基座验收通过；
- 官方 Mamba-2 完整支持 sequence 训练、在线 step、reset、snapshot/restore 和 checkpoint；
- Progress Expert 复用同一次冻结 prefix 的只读逐层 KV，并保持 Memory Token 隔离；
- 高噪声硬交接与现有损失在 PyTorch 中端到端可训练；
- Policy、WebSocket、LIBERO runner、实验矩阵与 profile 全部使用新 backend/schema；
- Mamba-3 只有在 RTX 5090 十项硬门全部通过后才标记可用；
- 旧 FutureMamba JAX 专用实现与含糊 backend 名称已删除；
- 定向测试、训练 smoke、真实闭环 smoke 与历史因果 smoke 均有可复现证据；
- checkpoint metadata 足以区分基座、Mamba 版本、kernel、dtype 与状态 schema。

若 Mamba-3 门控失败，但上述 Mamba-2 路径和干净切换均完成，则 FutureMamba 的纯 PyTorch 迁移已完成；Mamba-3 状态必须明确记录为当前硬件栈不支持，而不是伪装为已接入。

## 17. 参考来源

- OpenPI PyTorch 模型与训练：仓库内 `src/openpi/models_pytorch/`、`scripts/train_pytorch.py`。
- 官方 Mamba 仓库：https://github.com/state-spaces/mamba
- Mamba-2 固定版本：https://github.com/state-spaces/mamba/tree/v2.2.4
- Mamba-3 固定版本：https://github.com/state-spaces/mamba/tree/v2.3.2
- 官方 Mamba-3 模块：https://github.com/state-spaces/mamba/blob/v2.3.2/mamba_ssm/modules/mamba3.py
- 官方安装说明：https://github.com/state-spaces/mamba#installation
- OpenPI JAX → PyTorch 转换问题 #810：https://github.com/Physical-Intelligence/openpi/issues/810
- OpenPI LoRA 转换问题 #958：https://github.com/Physical-Intelligence/openpi/issues/958
- FutureMamba 方法设计：`docs/superpowers/specs/2026-08-08-futuremamba-design.md`。
