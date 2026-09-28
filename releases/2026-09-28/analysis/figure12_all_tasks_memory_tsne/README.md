# 十任务记忆 t-SNE：主实验高成功率修复版 checkpoint

## 当前状态

十个任务均已完成，共838个真实1024维记忆token，来自每任务固定5条示范。
A100 提取 T1、T2、T6、T7、T8、T10；LGD 直接加载原checkpoint提取 T3、T4、T5、T9，结果已回传本地。
LGD使用在A100按相同原始动作处理流程导出的 `replay_inputs.npz`，文件传输经SHA256校验。
**没有用旧版权重代替缺失任务。** 完整总览为 `fig12_all10_memory_tsne.pdf/png/svg`。
`fig12_PARTIAL_memory_tsne` 仅为此前保留的未完成预览，请使用完整总览。

## 选点来源

所有候选均来自 `../completed_experiments_source.json` 中经过审计的修复版主实验记录。
每任务、每训练seed、每checkpoint汇总三个evaluation seed，共60次测试。
选择成功数最高者；并列时依次取更早checkpoint、A100已有权重、seed42、较小seed。
这是一种基于已有评估结果的选择，供定性分析；不是新的无偏成功率估计。

|任务|训练seed|checkpoint|原评估成功数|权重位置|
|---|---:|---:|---:|---|
|T1|0|500|60/60|A100|
|T2|0|500|60/60|A100|
|T3|0|500|60/60|LGD|
|T4|1|500|60/60|LGD|
|T5|42|1000|60/60|LGD|
|T6|42|1500|48/60|A100|
|T7|0|2000|27/60|A100|
|T8|0|3000|34/60|A100|
|T9|0|500|60/60|LGD|
|T10|0|500|60/60|A100|

全部使用训练修复版 `differentiable_recurrent_step_v1`。最初考虑旧版的方案已取消。

## 提取与可视化

- 各任务固定取原始 HDF5 按数字顺序排列的前5条示范，不根据图形筛选。
- 对相应任务的真实训练权重进行离线因果重放；不是新策略rollout或独立泛化测试。
- 每20个执行动作查询一次，排除 q=0 的 learned empty-history token。
- 每个点来自1024维memory token。实际输入是经原管线处理的32维执行动作；关节角并非memory输入。
- 所有任务采用同样的降维规则：PCA最多30维，t-SNE perplexity=15、random_state=42、1500次迭代、PCA初始化、auto学习率。
- 每个任务独立拟合，轴坐标和图形整体距离不能跨任务比较。
- 颜色是该示范的归一化时间，不是标注的任务阶段；线表示同一条示范内的先后顺序。
- 不使用合成特征、时间拼接特征或人为构造聚类。图形好看与否不参与checkpoint或示范选择。

## 文件与复现

- `select_best_checkpoints.py`：从审计记录重建选择清单。
- `selected_checkpoints.json`、`selection_protocol.json`：原权重路径、三个评估记录与选择规则。
- `extract_all10_memory.py`：在原PhaseVLA运行环境提取，支持 `--tasks` 和 `--selection`。
- `extract_lgd_memory.py`：在LGD训练环境直接提取四个任务；模型参数严格加载，记录每个记忆参数张量的SHA256。
- `replay_inputs.npz`、`lgd_extraction_manifest.json`：传给LGD的处理后动作、原示范哈希与提取来源。
- `merge_lgd_features.py`：验证并合并LGD回传结果。
- `Txx_memory.npz`、`extraction_manifest.json`：真实记忆、query、episode、时间及模型校验值。
- `plot_all10_memory_tsne.py`：本地绘图脚本。
- `embeddings/`：缓存二维坐标与投影参数。
- `individual/`：各任务单独的PDF/PNG/SVG。

现有数据预览：

```powershell
python .\plot_all10_memory_tsne.py --available
```

重画完整十任务图：

```powershell
python .\plot_all10_memory_tsne.py
```

第二条命令在数据不齐全时会报错，不会静默生成冒充完整结果的图片。
