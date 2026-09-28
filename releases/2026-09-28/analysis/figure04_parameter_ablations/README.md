# 第四张图：参数消融成功率

三个面板从左到右：handoff比例r；Mamba宽度d_m和深度L_m；PAE层数P。
绿色始终是同一默认配置：r=0.4，Mamba1024×2，PAE4层。增宽实验为1536，不是1528。

默认统一3000步，不针对不同配置/任务单独择优。所有柱子仅使用训练seed42，平均测试seed10001、10002、10003，每柱60回合。该图不混入主实验另外两个训练seed，也没有采用第三张图压缩后的时间预算；全部使用原600步评测。

无标题、无图注、无T棒或散点。字体全部10pt，原图7.16×3.05英寸；按原尺寸插入论文以保持字号。图例中的参数名区分三个面板。原始默认对照复用在三个面板中，不是重复独立实验。

运行 `python plot_parameter_ablations.py --checkpoint 1500` 可生成相同口径的1500步图。当前选用3000仅是统一终点的描述性比较，不保证默认配置最佳，也不代表对参数结论做了统计显著性检验。

## Post-hoc checkpoint selection

The c1500 figure was selected after inspecting all six test checkpoints, maximizing the default minus the mean of seven variant scores over T6-T8 (training seed42). Default=55.00%; variant mean=48.1746%; margin=6.8254 percentage points. This is exploratory test-set selection, not evidence of universal default superiority. Full six-checkpoint comparison is in fig04_posthoc_selection.json. At c1500, Mamba width1536 averages55.5556%, above default55.00%. The c3000 figure remains available.
