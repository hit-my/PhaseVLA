# FutureMamba 研究设计规格

- **日期：** 2026-08-08
- **状态：** 架构、训练与实验设计已确认；仿真代码实现完成，正式训练、真实 LIBERO-Mem rollout 与真机实验待执行。
- **目标：** 为冻结的任务适配版 $\pi_{0.5}$ 增加参数高效的任务进程记忆插件，使其在当前观测相似、但已完成步骤不同的情况下选择正确动作，并减少重复执行。
- **主实验：** LIBERO-Mem；以 LIBERO-Long 检查一般能力保持；最后进行 4 类同构真机实验。
- **命名：** 整体方法暂称 **FutureMamba**；轻量动作模块正式称 **Progress Expert**，不称 Future Expert。

> 投稿说明：当前设计遵循机器人会议论文的证据标准。会议与年份需要按实际投稿窗口重新确认；本文档不把投稿时间作为方法设计的一部分。

## 1. 问题定义

### 1.1 目标失效模式

标准 action-chunked VLA 在第 $q$ 次策略查询时，根据当前观测 $o_q$、本体状态 $r_q$ 和指令 $\ell$ 生成动作块。若策略只依赖当前输入，则隐含采用马尔可夫假设：

$$
\pi(a_q\mid o_{1:q},a_{1:q-1},r_{1:q},\ell)
\approx
\pi(a_q\mid o_q,r_q,\ell).
$$

FutureMamba 关注该假设失效的场景。存在两段不同历史 $\mathcal H_q^1$ 和 $\mathcal H_q^2$，它们产生相同或近似的当前输入，但正确动作分支不同：

$$
(o_q^1,r_q^1,\ell)\simeq(o_q^2,r_q^2,\ell),
\qquad
\mathcal A^*(\mathcal H_q^1)\cap\mathcal A^*(\mathcal H_q^2)=\varnothing.
$$

论文聚焦两类可观测失败：

1. **阶段消歧：** 当前画面相似，但因已完成步骤不同，下一动作应不同。
2. **完成状态与防重复：** 策略应记住已完成子目标，避免重复抓取、重复放置或超过指定次数。

### 1.2 研究假设

核心假设不是「Mamba 本身能提高 VLA」，而是：

> 在历史混淆状态中，任务进程主要决定 action flow 的全局动作模态；当前观测与本体状态主要决定低噪声阶段的精细运动。因此，记忆插件只需接管高噪声的早期 flow integration，冻结的 $\pi_{0.5}$ Action Expert 可以继续负责低噪声精修。

该假设对应一个可证伪的噪声时间分工：

- **高噪声阶段：** Progress Expert 根据当前视觉语言上下文与跨查询进程记忆选择「接下来做什么」。
- **低噪声阶段：** 冻结 Action Expert 根据当前几何与本体状态细化「具体怎么做」。

### 1.3 范围边界

本工作解决标准闭环执行中的任务进程追踪，不承诺解决以下问题：

- 未被当前观测、本体状态或已执行动作暴露的隐蔽执行失败；
- 开放世界语义记忆、跨 episode 终身记忆或外部知识检索；
- 显式高层任务规划、语言子目标生成或未来图像预测；
- 依赖未执行 action tail 的跨查询运动连续性；
- 未经适配的机器人 embodiment、动作空间或低层控制技能迁移。

## 2. 与相关工作的创新边界

「增加历史记忆」不能单独作为贡献。以下工作已经覆盖相邻方向：

- **MemoryVLA：** 使用 EOS 位置认知 Token、感知—认知记忆库，以及记忆条件 diffusion action expert。其记忆通过检索、融合和合并支持完整动作生成。
- **DSSP：** 使用 Mamba/SSM 压缩完整观测历史，以未来状态辅助目标塑造记忆，并将历史上下文作为前缀条件输入 SSM 去噪器。
- **ChainVLA：** 显式维护跨 action-chunk 查询的 Progress Context 和 Motion Tail，以统一执行状态连接连续查询。
- **LIBERO-Mem / Embodied-SlotSSM：** 已将相似观测、不同对象历史、重复次数、关系与遮挡定义为非马尔可夫操作基准。

FutureMamba 的主创新边界限定为：

1. **冻结 VLA 的查询级进程状态：** 使用标准 Mamba 增量状态压缩当前 VLM 语义与上一周期实际执行动作，不修改任务适配后的 $\pi_{0.5}$ 参数。
2. **沿 flow-matching 时间轴的专家分工：** Progress Expert 直接替代前 $K$ 个高噪声求解步的速度场；冻结 Action Expert 从交接状态继续剩余求解步。
3. **硬交接对齐：** 在标准高噪声 flow-matching 目标外，约束 Progress Expert 滚动后的交接状态落在真实条件概率路径上。
4. **参数高效且可证伪的验证：** 用等参数、全程条件、无记忆、软融合和不同交接比例证明收益来自进程记忆与噪声时间分工，而非单纯增加参数。

不应在论文中声称以下内容：

- 首次使用 Mamba 进行机器人历史编码；
- 首次让记忆条件化动作去噪器；
- 首次维护跨 action chunk 的任务进程；
- Progress Expert 预测未来状态。它预测的是未来动作块在高噪声区间的 flow velocity。

## 3. 系统架构

### 3.1 两阶段训练与冻结边界

采用任务适配后的 $\pi_{0.5}$ checkpoint，而不是未经目标机器人适配的通用 checkpoint。

**阶段 A：任务适配基座。**

- 使用标准 $\pi_{0.5}$ 训练目标，在目标 embodiment 和任务训练集上适配 VLM/Action Expert。
- 训练输入保持基座原有的单次查询形式，不引入 FutureMamba 记忆。
- LIBERO 实验中，基座在 LIBERO-Mem 与 LIBERO-Long 的训练演示上适配；两类数据按 episode 平衡采样。
- 真机实验单独在 4 类真机任务的训练演示上适配，不把仿真到真机零样本迁移作为本工作的必要结论。

**阶段 B：训练插件。**

冻结以下参数并全程 `stop-gradient`：

- $\pi_{0.5}$ 视觉编码器；
- $\pi_{0.5}$ VLM；
- $\pi_{0.5}$ Action Expert；
- 基座动作输入、时间条件和动作输出投影。

只训练：

- VLM Token 输入投影 $W_z$；
- 已执行动作编码器 $E_a$；
- Mamba 进程记忆；
- Memory Token 投影 $W_m$；
- Progress Expert 及其独立输入/输出接口。

若插件可训练参数超过冻结 $\pi_{0.5}$ 总参数的 10%，论文不得使用「lightweight」作为无条件结论；必须报告实际参数量与 FLOPs。

### 3.2 每个 Action Chunk 的输入

第 $q$ 次策略查询由当前观测、本体状态与指令构成：

$$
x_q^{\mathrm{obs}}=(o_q,r_q,\ell).
$$

冻结 VLM 输出最后一层条件序列：

$$
H_q=\operatorname{VLM}_{\mathrm{frozen}}(o_q,r_q,\ell).
$$

记忆输入使用现有语言序列的**最后一个有效 Token**，而不是 padding 后的固定末位：

$$
z_q=H_q[p_q],
\qquad
p_q=\max\{j:\text{prompt-mask}_{q,j}=1\}.
$$

该位置可对应 EOS 或最后一个有效 prompt token，具体取决于 tokenizer；不向冻结 VLM 新增 Memory Token。

### 3.3 已执行动作摘要

记忆不仅接收当前 VLM Token，还接收上一策略周期真正下发执行的 action prefix：

$$
A_{q-1}^{\mathrm{exec}}=(a_{q-1,1},\ldots,a_{q-1,h_{\mathrm{exec}}}).
$$

动作编码器采用固定、轻量的定义：

1. 对每个已执行动作应用共享两层 MLP；
2. 对有效动作位置做 masked mean pooling；
3. 将池化结果与最后一个已执行动作的编码拼接；
4. 通过线性层得到动作摘要 $\bar a_{q-1}^{\mathrm{exec}}$。

第一个查询没有历史动作，令 $\bar a_{-1}^{\mathrm{exec}}=0$。只输入实际执行的 prefix，不输入上一动作块中未执行的 tail。

记忆步输入为：

$$
e_q=W_e\left[W_z z_q\,\Vert\,E_a(A_{q-1}^{\mathrm{exec}})\right].
$$

训练时使用演示轨迹中真实执行的动作，并对动作摘要加入与基座闭环误差同量级的零均值扰动，降低 teacher-forcing 与部署动作之间的分布差异。该扰动只作用于记忆输入，不修改动作监督标签。

### 3.4 标准 Mamba 增量记忆

FutureMamba 使用标准多层 Mamba 的增量推理状态，而不是仅维护一个普通 RNN 隐向量：

$$
(h_q,S_q)=\operatorname{MambaStep}_{\phi}(e_q,S_{q-1}).
$$

其中：

- $S_q$ 是跨 Action Chunk 持久化的 memory state；
- $S_q$ 包含每层 selective SSM state 与 causal-convolution cache；
- $h_q$ 是第 $q$ 步最后一层输出，仅用于构造当前 Memory Token；
- episode 开始时 $S_0=0$，结束或环境 reset 时立即清空；
- 状态大小不随 episode 长度增长，每次查询的增量计算量相对历史长度为 $O(1)$。

Memory Token 为：

$$
m_q=W_m h_q.
$$

训练时可对完整 query 序列使用等价的 causal selective scan；推理时使用逐查询 `MambaStep`。训练与推理必须采用相同的因果方向和 episode reset 规则。

### 3.5 Progress Expert 与共享冻结前缀缓存

Progress Expert 与 $\pi_{0.5}$ Action Expert 使用相同类型的 Transformer block、动作 Token 接口和时间条件方式，但采用更少层数：

$$
L_P\approx\frac{1}{4}L_A,
\qquad
d_P=d_A.
$$

主设置为同宽、约 1/4 深度；实验消融 1/8、1/4 和 1/2 深度。若 $L_A$ 不能被 4 整除，取最接近且不少于 1 层的整数。「结构相同」只表示 block 类型与输入输出合同相同，不表示复制完整 300M Action Expert。

冻结 VLM 对当前图像、语言和本体状态只执行一次 prefix forward，并返回最后层条件序列以及逐层前缀 KV Cache：

$$
(C_q,\mathcal K_q^C,\mathcal V_q^C)=\operatorname{VLM}_{\mathrm{frozen}}(o_q,r_q,\ell).
$$

Action Expert 的 $L_A$ 层继续使用这份原始缓存。Progress Expert 不重新运行 VLM，也不学习另一套当前上下文 K/V 投影；其第 $j$ 层直接读取同一份冻结缓存的第 $r(j)$ 层，其中 $r(j)$ 在 $[1,L_A]$ 上等间隔取 $L_P$ 个索引，并随配置落盘。这样两位专家看到完全相同的当前场景与指令条件，区别只来自 Progress Expert 的历史记忆及其可训练动作侧参数。

Memory Token 不写回冻结 VLM，也不进入 Action Expert。Progress Expert 为每层单独把 $m_q$ 投影为 $(K_{q,j}^M,V_{q,j}^M)$，并追加到该层冻结前缀缓存：

$$
(K_{q,j}^P,V_{q,j}^P)=
[(K_{q,r(j)}^C,V_{q,r(j)}^C);(K_{q,j}^M,V_{q,j}^M)].
$$

Progress action query 随后对共享当前上下文、Memory Token 和本层 action tokens 做注意力：

$$
v_P=f_P(x_t,t,\mathcal K_q^C,\mathcal V_q^C,m_q).
$$

注意力合同必须满足：

- 两位专家复用同一次冻结 prefix forward 的同一份逐层 VLM KV Cache；
- Progress Expert 的 action tokens 可以注意共享前缀 KV 与 $m_q$；
- Memory Token 只生成 Progress Expert 的额外 K/V，不进入冻结 VLM 自注意力；
- 冻结 Action Expert 始终只读取原始前缀 KV，不读取 $m_q$；
- `use_prefix_cache=False` 只用于「Memory without current context」消融，不能作为主方法。

### 3.6 高噪声硬切换

沿用官方 openpi 实现的时间约定：

- $t=1$ 为高斯噪声；
- $t=0$ 为动作数据；
- 采样从 1 向 0 积分；
- 训练插值与目标速度为

$$
x_t=t\epsilon+(1-t)a,
\qquad
u=\epsilon-a.
$$

设总求解步数为 $N$，交接比例为 $\rho$：

$$
K=\lceil \rho N\rceil,
\qquad
t_i=1-\frac{i}{N},
\qquad\Delta t=-\frac{1}{N}.
$$

初始化 $x_0=\epsilon\sim\mathcal N(0,I)$。前 $K$ 个高噪声步只运行 Progress Expert：

$$
x_{i+1}=x_i+\Delta t\,f_P(x_i,t_i,C_q^P),
\qquad 0\le i<K.
$$

剩余求解步只运行冻结 Action Expert：

$$
x_{i+1}=x_i+\Delta t\,f_A(x_i,t_i,C_q),
\qquad K\le i<N.
$$

最终动作块为 $\hat A_q=x_N$。

该机制是**直接替代与状态交接**，不是把 $v_P$ 输入 Action Expert。Progress Expert 通过更新后的 $x_K$ 间接影响后半程；Action Expert 从 $x_K$ 接管。

主设置从 $N=10$、$\rho\in\{0.2,0.3\}$ 中根据验证集预注册选择一个，不在测试集调参。完整报告 $\rho\in\{0,0.1,0.2,0.3,0.5,1.0\}$。

## 4. 训练目标

### 4.1 完整 Episode 展开

插件训练保持 episode 顺序。对每条包含 $Q$ 个 Action Chunk 的演示：

$$
S_0=0,
$$

$$
(h_q,S_q)=\operatorname{MambaStep}(e_q,S_{q-1}),
\qquad q=1,\ldots,Q.
$$

所有查询损失之和对整个 episode 反向传播，不在 Action Chunk 边界截断 $S_q$ 的梯度。由于 VLM 已冻结，$H_q$ 与 $C_q$ 可离线缓存；训练图只保留插件部分。

为避免长短 episode 权重失衡，先对每条 episode 内的 query 损失取均值，再对 batch 内 episode 取均值。

### 4.2 高噪声 Flow-Matching 损失

按官方 $\pi_{0.5}$ 分布采样 flow 时间，但只保留高噪声交接区间：

$$
t\sim p_{\pi_{0.5}}(t\mid t\ge t_K),
\qquad t_K=1-\frac{K}{N}.
$$

给定真实动作块 $a_q$ 与噪声 $\epsilon_q$：

$$
x_{q,t}=t\epsilon_q+(1-t)a_q,
\qquad
u_q=\epsilon_q-a_q.
$$

Progress Expert 的局部监督为：

$$
\mathcal L_{\mathrm{FM}}^{(q)}
=
\mathbb E_t
\left[
\left\|f_P(x_{q,t},t,C_q^P)-u_q\right\|_2^2
\right].
$$

### 4.3 交接状态与边界速度对齐

仅使用局部 flow-matching 会造成 teacher-forced 状态与推理滚动状态不一致。为此，从同一噪声 $\epsilon_q$ 出发，用 Progress Expert 按实际前 $K$ 个 solver step 滚动得到 $\hat x_{q,t_K}$。

真实线性概率路径在交接时间的状态为：

$$
x^*_{q,t_K}=t_K\epsilon_q+(1-t_K)a_q.
$$

交接状态损失为：

$$
\mathcal L_{\mathrm{handoff}}^{(q)}
:=
\left\|\hat x_{q,t_K}-x^*_{q,t_K}\right\|_2^2.
$$

状态落在合理中间噪声分布仍不能保证两个速度场在切换点连续。令 $\bar x_{q,t_K}=\operatorname{stopgrad}(\hat x_{q,t_K})$，边界速度损失为：

$$
\mathcal L_{\mathrm{boundary}}^{(q)}
:=
\left\|
f_P(\bar x_{q,t_K},t_K,\mathcal K_q^C,\mathcal V_q^C,m_q)
-
\operatorname{stopgrad}\left[f_A(\bar x_{q,t_K},t_K,\mathcal K_q^C,\mathcal V_q^C)\right]
\right\|_2^2.
$$

对交接状态停止梯度可避免模型通过移动 $\hat x_{q,t_K}$ 人为降低边界项；冻结 Action Expert 只提供目标速度，不接收梯度。

总目标：

$$
\mathcal L
:=
\frac{1}{Q}\sum_{q=1}^{Q}
\left(
\mathcal L_{\mathrm{FM}}^{(q)}
+
\lambda_h\mathcal L_{\mathrm{handoff}}^{(q)}
+
\lambda_b\mathcal L_{\mathrm{boundary}}^{(q)}
\right).
$$

$\lambda_h$ 与 $\lambda_b$ 只在训练集与验证集选择；分别报告置零消融。若边界项抑制了由记忆引起的必要行为分歧，则主设置必须改为 $\lambda_b=0$，不能为了叙事保留无效约束。核心方案不增加未来观测预测、阶段分类或 oracle 子目标标签。

### 4.4 训练完整性约束

- VLM 与 Action Expert 必须处于 eval/frozen 状态；不能依赖遗漏到 optimizer 之外的「伪冻结」。
- 每个 batch 必须保留 episode ID、query 顺序、有效 query mask 和 reset mask。
- 不允许把验证或测试轨迹前缀用于训练 Mamba 状态。
- 随机裁剪只能用于显存诊断；主结果使用完整 episode 展开。
- 若完整展开无法容纳，备选方案是带无梯度 burn-in 的截断反传，但必须单列为方法变体，不能与主结果混用。

## 5. 实验设计

### 5.1 仿真基准

**主基准：LIBERO-Mem。**

使用其 10 个非马尔可夫任务。官方固定提交的 RLDS builder 只公开 `train` split，因此实验采用预注册、按数值 demo ID 排序的本地训练/验证拆分，并将 split manifest 与源码 commit 随结果发布。任务覆盖：

- Object Motion；
- Object Sequence，包括重复 3、5、7 次；
- Object Relation；
- Object Occlusion。

LIBERO-Mem 提供对象 ID、mask 和子目标完成标记，可用于阶段级评测与受控历史干预。

**能力保持：LIBERO-Long。**

LIBERO-Long 不作为记忆因果性的主证据。它用于检查插件是否破坏任务适配版 $\pi_{0.5}$ 已有的一般长程操作能力。

### 5.2 主要指标

1. **Task Success Rate：** 完整任务成功次数除以总 rollout 数。
2. **Subgoal Completion：** 按 LIBERO-Mem 官方 Sequence / Or 子目标定义计算完成比例。
3. **Redundant Execution Rate：** 使用环境符号事件监控器，统计已完成子目标被再次触发，或计数任务超过目标次数后再次启动同类操作的 Action Chunk 数，占全部可判定决策 Chunk 的比例。
4. **History-Conditioned Branch Accuracy：** 在相同当前输入、不同有效历史的配对测试中，首个执行动作前缀是否进入对应历史的正确分支。
5. **Temporal Scaling：** 随重复次数、关系链长度和遮挡持续时间增加，报告成功率与子目标完成率曲线。
6. **Efficiency：** 可训练参数、总参数、每次查询 FLOPs、端到端延迟、峰值显存、Mamba state 大小。
7. **Capability Retention：** FutureMamba 相对冻结基座在 LIBERO-Long 上的绝对成功率变化。

所有主要成功率报告每任务固定 rollout 数、至少 3 个训练种子、均值与 95% 置信区间。若机器人资源不足以支持 3 个真机训练种子，真机部分至少报告固定 trial 数、Wilson 区间和所有失败类别，不把单次最好结果作为主表数字。

### 5.3 历史因果干预

为证明性能来自记忆，而非当前图像中的微小线索，构造成对反事实测试：

1. 选择同一任务中进程不同、但当前 simulator state 可恢复为相同配置的两段有效历史；
2. 分别沿两段历史更新得到 $S_q^1$ 与 $S_q^2$；
3. 在决策点向两次推理提供相同的当前 RGB、本体状态、指令和高斯噪声 seed；
4. 唯一变化为 Mamba state；
5. 判断首个执行动作前缀是否分别进入正确且不同的专家动作分支；
6. 交换 $S_q^1$ 与 $S_q^2$，检查动作选择是否随进程状态交换。

该测试不能只比较 action L2。应使用环境分支事件、短程闭环结果或到各专家分支的归一化距离判定动作语义。

### 5.4 基线

最低必要基线：

1. **Frozen task-adapted $\pi_{0.5}$：** 无历史、无插件。
2. **Recent Frame Stack：** 固定最近帧窗口，参数与训练数据公平。
3. **GRU/LSTM Progress Memory：** 用 GRU 或 LSTM 替换 Mamba，其余 Progress Expert 与交接机制不变，并匹配可训练参数。
4. **Progress Expert + shared prefix KV, without Memory：** 保留共享当前上下文与早期轻量专家，但令 $m_q=0$；排除「只是增加了一个专家」的解释。
5. **Progress Expert + Memory, without prefix KV：** 只给历史记忆、noisy action 与 flow 时间；检验当前指令和场景语义是否仍然必要。
6. **Frozen Action Expert + Memory, full horizon：** 不增加早期专家；为冻结 Action Expert 每层追加可训练的 Memory K/V，并在全部 solver step 条件化。它检验「早期接力」是否优于常规全程记忆条件化。
7. **Mamba Reset Every Query：** 每个 Action Chunk 将 $S_q$ 清零；检验跨查询状态是否必要。
8. **Shuffled History：** 在同任务 episode 内打乱历史 Token/动作对；检验时间顺序是否必要。
9. **Full-Horizon Progress Expert：** $\rho=1$；检验冻结 Action Expert 的低噪声精修是否必要。
10. **No Progress Expert：** $\rho=0$；等价于冻结基座。
11. **Oracle Progress Condition：** 只在诊断实验中输入官方子目标进度，作为阶段可辨识性的上限；不得计入可部署方法。

若 LaMem-VLA、EventVLA、MemoryVAM、MemoryVLA、DSSP、ChainVLA 或 Embodied-SlotSSM 的官方实现能在相同 benchmark、动作空间、训练数据和 backbone 条件下运行，则增加直接对照。否则只做定性相关工作比较，不能跨数据集直接排列不公平数字。

### 5.5 机制消融

- $\rho\in\{0,0.1,0.2,0.3,0.5,1.0\}$，并在 $N=10$ 时显式报告对应 $K\in\{0,1,2,3,5,10\}$；
- 硬切换、完整速度场凸融合、残差修正融合；
- Progress Expert 深度比例 1/8、1/4、1/2；
- 共享冻结 prefix KV、无 prefix KV，以及无 Memory Token；
- 有/无 $\mathcal L_{\mathrm{handoff}}$，有/无 $\mathcal L_{\mathrm{boundary}}$；
- 仅 VLM Token，与 VLM Token + 已执行动作；
- 记忆表示：末端有效 Token、最后层注意力池化、4 个与 8 个 learned-query 压缩 Token；
- 对象标注只做 oracle object-aware 上限，不作为可部署输入；若实现 mask-free object slots，则单列为方法变体；
- 标准 Mamba state，与参数匹配 GRU/LSTM；
- 完整 episode 反传，与截断反传；
- 正确历史、清零历史、截断历史、打乱历史、交换历史；
- 同参数量但随机初始化的早期专家，排除参数量解释。

消融主结论必须同时看 Task Success、Branch Accuracy 和 Redundant Execution Rate。只提升动作重建误差不足以支持任务进程主张。

## 6. 真机实验

真机任务与 LIBERO-Mem 的 4 类记忆需求同构，每类至少 1 个任务：

1. **运动/计数：** 将同一物体拿起并放回指定次数，要求达到次数后停止。
2. **序列：** 按指令顺序操作 3 个视觉相似物体。
3. **关系：** 根据此前使用过的对象或容器决定后续放置目标。
4. **遮挡：** 将物体放入容器并关闭，随后根据历史选择正确容器或对象。

每类任务包含两种 trial：

- **自然闭环 trial：** 从任务开始执行至成功或超时；
- **历史配对 trial：** 尽量恢复相同当前布局，但通过不同有效历史形成不同 Mamba state，测试下一动作分支。

真机报告：

- 每任务 trial 数；
- 成功率与置信区间；
- 子目标完成率；
- 重复执行次数；
- 阶段选择错误、感知错误、抓取错误、控制错误和不可恢复错误的分类；
- 冻结基座与 FutureMamba 的成对对照；
- 推理延迟是否满足控制频率。

不得把「基础抓取失败」记为记忆失败，也不得在只展示成功视频时省略完整 trial 统计。

## 7. 论文主张与证伪标准

### 7.1 可支持的主张

若实验满足本节条件，论文可以主张：

1. 固定大小的查询级 Mamba state 能在冻结 $\pi_{0.5}$ 上编码任务进程，并改善阶段消歧与防重复。
2. 对历史依赖任务，只在 flow-matching 高噪声阶段使用 Progress Expert，优于不使用记忆、全程使用轻量专家或参数匹配的当前观测专家。
3. 交接对齐降低 Progress Expert 与冻结 Action Expert 之间的 solver-state 分布偏移。
4. 插件在提高 LIBERO-Mem 与同构真机任务表现的同时，基本保持 LIBERO-Long 能力，并以固定历史状态大小运行。

### 7.2 必须通过的证伪标准

任一条件不满足，都应收缩对应结论：

- 清零、打乱或交换历史后，记忆收益应显著下降；
- 历史配对测试中，相同当前输入与不同 $S_q$ 应产生对应的正确动作分支；
- 合适的早期 $0<\rho<1$ 应优于 $\rho=0$ 和 $\rho=1$；
- 同参数量、无记忆的 Progress Expert 不应达到完整方法的增益；
- 去掉交接对齐应在交接误差或下游成功率上造成可测退化，否则不能声称该损失必要；
- LIBERO-Long 与真机基础操作不能出现未解释的显著退化；
- 若软融合稳定优于硬切换，则主方法应改为数据支持的融合版本，不能维护先验叙事；
- 若 Progress Expert 只改善短程动作质量而不改善 Branch Accuracy 或重复执行率，则不能把收益归因于任务进程记忆；
- 若仅增加参数即可取得相同收益，则噪声时间分工的贡献不成立。

## 8. 主要风险与处理

### 8.1 与近邻工作的重合

**风险：** Mamba 历史编码、末端认知 Token 和记忆条件动作生成均已有先例。

**处理：** 所有方法表述围绕高噪声硬切换、冻结基座、交接对齐与因果消融展开；不把通用记忆模块包装成首创。

截至 2026 年 8 月，最接近的工作已覆盖「VLA + memory」这一宽泛命题：

- **LaMem-VLA** 把短期视觉记忆与长期任务进程/动作连续性记忆压缩为 latent tokens，并在动作形成前写入 VLA 原生推理序列；
- **EventVLA** 用初始帧、短期历史和事件驱动关键帧保存稀疏视觉证据，并专门评测中途出现后消失的信息；
- **MemoryVAM** 把 episode memory 同时注入未来预测 backbone 与 action decoder，并在 LIBERO-Mem 报告从 5% 到 42.5% 的平均成功率提升；
- **DynaGuide** 用外部 dynamics model 的目标梯度在去噪过程中修改基础 policy 的方向，属于 inference-time steering，而不是从在线历史学习高噪声 velocity field 后交给冻结专家。

因此本文不能把「用 Mamba 给 VLA 加记忆」作为创新主张。可检验差异必须限定为：**记忆在 flow trajectory 的哪个噪声阶段介入**，以及 history-conditioned Progress Expert 是否只负责高噪声行为模式选择、随后把连续 latent 原样交给冻结 Action Expert 完成精细执行。若早期接力不优于全程条件化或融合，该创新主张不成立。

### 8.2 交接分布偏移

**风险：** 冻结 Action Expert 在训练时见到解析 $x_t$，推理时却接收 Progress Expert 产生的 $\hat x_{t_K}$。

**处理：** 使用多步 rollout 的 $\mathcal L_{\mathrm{handoff}}$；报告交接状态误差；消融硬切换、凸融合与残差融合。

### 8.3 单个末端 Token 的信息瓶颈

**风险：** 单个全局末端 Token 可能无法区分多个外观相同物体；LIBERO-Mem 的重复计数、交换顺序和遮挡任务会放大该瓶颈。

**处理：** 当前完整 VLM 前缀 KV 仍直接输入 Progress Expert，单个 Mamba 输入只承担跨查询进程。首个可运行版本保留末端有效 Token；论文主实验必须比较注意力池化与 4/8 个 learned-query 压缩 Token。官方对象 mask 只做 oracle 上限，避免把不可部署标注泄漏进主方法。若多 Token 明显优于单 Token，则论文方法定义随证据升级，不能继续把单向量当作充分表示。

### 8.4 动作历史的部署偏移

**风险：** 训练读取演示动作，部署读取模型实际动作。

**处理：** 记忆输入使用实际执行 prefix；训练加入动作扰动；报告无动作输入消融。若闭环偏移仍显著，再增加 simulator rollout 数据，但不作为首版方法的隐含前提。

### 8.5 进程与失败恢复混淆

**风险：** 仅凭动作命令无法证明动作成功。

**处理：** 方法只主张标准执行中的进程追踪。失败恢复仅在当前观测可反映失败时评测；不可观测执行失败明确列为范围外问题。

### 8.6 「轻量」缺乏量化

**风险：** 同宽 1/4 深度 Expert 仍可能带来较多参数。

**处理：** 报告插件参数、FLOPs、延迟与显存；以冻结 VLA 总参数占比和端到端查询成本定义轻量，不只按层数描述。

## 9. 实现验收标准

实现进入正式实验前必须满足：

- 同一 episode 连续查询后 $S_q$ 发生更新，reset 后与零状态完全一致；
- 改变 padding 长度不改变所选末端有效 Token；
- Progress Expert 的 Memory Token 不进入冻结 VLM 或 Action Expert；
- $\rho=0$ 的输出与冻结基座在相同噪声 seed 下数值一致；
- $\rho=1$ 全程不调用 Action Expert；
- 硬切换时第 $K$ 步的 $x_K$ 原样交给 Action Expert，不做未声明的重采样；
- 训练与推理均采用 $t=1\rightarrow0$ 的官方 openpi 时间方向；
- 完整 episode 展开不跨 episode 传播状态；
- 冻结参数训练前后校验和不变；
- 历史配对测试使用相同当前输入与相同噪声 seed；
- 报告的重复执行率可由环境事件日志自动复算；
- 所有主表数字绑定固定 checkpoint、配置、随机种子与原始 rollout 日志。

## 10. 预期论文结构

1. **Introduction：** 当前观测无法判断任务进程；将问题收敛到阶段消歧和防重复。
2. **Related Work：** 记忆 VLA、SSM/Mamba 历史编码、action-chunk 连续性、diffusion/flow expert 分工。
3. **Method：** 查询级 Progress Memory、Progress Expert、高噪声硬切换、交接对齐。
4. **Experimental Setup：** task-adapted frozen $\pi_{0.5}$、LIBERO-Mem、LIBERO-Long、真机任务、历史干预协议。
5. **Results：** 主结果、历史因果测试、时间扩展、效率、能力保持。
6. **Ablations：** $\rho$、Expert 深度、Mamba/GRU/LSTM、历史破坏、交接损失、硬切换/融合。
7. **Limitations：** 不可观测执行失败、对象级细节瓶颈、真机数据规模与近邻工作重合边界。

## 11. 参考资源

- [$\pi_0$ / $\pi_{0.5}$ 官方 openpi 实现](https://github.com/Physical-Intelligence/openpi)
- [LIBERO-Mem 项目页、数据与代码](https://libero-mem.github.io/)
- [Rethinking Progression of Memory State in Robotic Manipulation / LIBERO-Mem](https://ojs.aaai.org/index.php/AAAI/article/download/37337/41299)
- [LaMem-VLA: Dual Latent Memory in Vision-Language-Action Models](https://arxiv.org/abs/2607.07608)
- [EventVLA: Event-Driven Visual Evidence Memory](https://arxiv.org/abs/2606.20092)
- [MemoryVAM: Integrating Memory into Video Action Model](https://arxiv.org/abs/2606.20679)
- [DynaGuide: Steering Diffusion Policies with Active Dynamic Guidance](https://arxiv.org/abs/2506.13922)
- [MemoryVLA: Perceptual-Cognitive Memory in Vision-Language-Action Models](https://arxiv.org/abs/2508.19236)
- [DSSP: Diffusion State Space Policy with Full-History Encoding](https://arxiv.org/abs/2605.14598)
- [ChainVLA: Chaining VLA Queries through a Unified Execution State](https://arxiv.org/abs/2608.02326)
- [RoboMemArena: A Comprehensive and Challenging Robotic Memory Benchmark](https://arxiv.org/abs/2605.10921)

## 12. 已确认决策记录

- 核心失效模式：阶段消歧 + 完成状态与防重复。
- 记忆更新频率：每个 Action Chunk 查询一次，而非每个底层控制 step。
- 记忆输入：首版使用 VLM 最后一层末端有效 Token + 上一周期实际执行动作 prefix；论文消融 4/8 个 learned-query 压缩 Token。
- Mamba 定义：标准增量 Mamba state，包括 SSM state 与卷积 cache；episode reset 时清零。
- 专家耦合：Progress Expert 前 $K$ 步直接替代，Action Expert 后续接管；凸融合与残差融合只作消融/备用。
- 当前条件：冻结 VLM prefix 只计算一次；Progress Expert 与 Action Expert 复用同一份逐层 KV Cache。
- Memory Token 位置：只向 Progress Expert 追加逐层 Memory K/V，不进入冻结 VLM 或 Action Expert。
- Progress Expert 规模：与 Action Expert 同宽、同 block 类型，约 1/4 深度，不宣称复制完整 300M 参数规模。
- 训练目标：高噪声真值 Flow Matching + 交接状态对齐 + 边界速度对齐；后两项分别消融。
- 反向传播：完整 episode 跨 Action Chunk 展开。
- 切换点：$K=\lceil\rho N\rceil$；初期测试 $K\in\{1,2,3,5\}$，完整报告 $K\in\{0,1,2,3,5,10\}$。
- 基座：冻结任务适配后的 $\pi_{0.5}$。
- 主评测：LIBERO-Mem；LIBERO-Long 检查能力保持；4 类同构真机任务。
- 创新主轴：记忆介入 action flow 的噪声阶段与硬交接，不是 Mamba 或通用 VLA memory 本身。
- 轻量专家正式名称：Progress Expert。
