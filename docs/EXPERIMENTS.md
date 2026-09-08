# 实验范围与运行方法

## 冻结范围

数值实验沿用`e0_v1_4.json`与来源修订后的归档结果；EuRoC沿用v1.5科学配置和v1.6收窄范围。公开适配不更改模型、数据生成、随机种子、方法、指标或门槛，不新增实验。

EuRoC比较Raw、Global-DoF、DPR-Exact-Block、DUC-K16和Permuted-K16。拟合集N=16和N=48均须运行，留出数据用于评价，不参与拟合。所有方法沿用相同候选帧对与原有共同有效样本规则。

六条序列为`V1_01_easy`、`V1_02_medium`、`V1_03_difficult`、`V2_01_easy`、`V2_02_medium`、`V2_03_difficult`。不能只运行表现较好的序列或删除未通过的帧对记录。

## 论文可支持的结论

N=16的五条主要序列共同有效帧对数依次为265、128、117、204和180。每条序列先汇总帧对，再对序列等权汇总；不是把894个相关帧对当成894个独立样本。DUC-K16相对Raw的序列等权中位差为：CE95为−0.025，绝对对数尺度误差约−0.06696，留出NLL约−0.21022。负值表示误差降低。

N=16的校正增量时间中位数：DUC-K16约0.1779 ms，精确块校正约0.4619 ms；共享QR计算另计，约0.0385 ms。61.5%的降低仅对应这两个校正路径的增量时间，不是系统速度提升。

以下边界与有利结果共同构成最小证据集：

- N=48的上述三项差值分别为0、约+0.03815和+0.02943，不支持同样的校准收益。
- V2_03在两种拟合集大小下分别只有58和57个共同有效帧对，低于每序列80个的门槛。原六序列门槛未通过的记录保持不变；v1.6是继承原结果后的收窄分析，不是一次新的独立确认实验。
- 非线性尺度误差点估计降低17.67%，但配对bootstrap差值的单侧95%上界约+0.004139，不能据此宣称明确的统计优势。
- 置换对照也在部分指标上有利，因此不能把校准改善解释为唯一或已证实的作用机制。
- 顺序实验使用精确标量块迹校正，不等同于随机近似已通过闭环VIO验证；边界命中结果必须保留。

## 重新运行数值实验

以下命令是完整实验入口，不属于默认快速检查。本次发布只执行了单元测试和最小样本检查，没有重新运行这些完整实验。

```bash
python scripts/run_e0.py --profile full --kind linear --output outputs/e0-linear
python scripts/run_e0.py --profile full --kind nonlinear --output outputs/e0-nonlinear
python scripts/run_e0.py --profile full --kind sequential --output outputs/e0-sequential
```

可按固定实例编号分片，例如`--shards 4 --shard-index 0`；其余分片分别用1、2、3且使用不同输出目录。合并前必须核对编号无重复、无缺失、总数与冻结规模一致。分片不会改变实例种子。

输出为逐问题JSONL和运行回执。新输出标记为独立复现、待审查，绝不冒充原已审核结果。数值失败记录继续写入输出并使命令返回非零退出码；科学效果不佳不是程序故障。不要通过改变随机种子或删除失败实例来获得有利结果。

v1.0.1会在启动前拒绝非法类别、重复类别和空分片。运行回执区分`COMPLETED`、`COMPLETED_WITH_NUMERICAL_FAILURES`及`INCOMPLETE_EXCEPTION`；遇到可捕获异常时保留已写入行、计数和文件哈希。进程被强制终止等无法捕获的中断仍需核对逐行结果，不能把遗留的`RUNNING`状态当成完成。回执不代表新的论文证据自动通过审查。

完整运行沿用内核的冻结样本数与参数；`smoke`只缩小问题规模并明确标记为测试。内核中的逐问题计时不等于原服务器独立预热后的计时基准；新运行需在同一机器、相同线程设置下独立评价，不能直接替换历史记录中的时间。

## 重新运行EuRoC

使用已有的官方ASL目录结构，`--dataset-root`下面直接放置六个序列目录：

```text
EUROC_ROOT/
  V1_01_easy/mav0/cam0/{sensor.yaml,data.csv,data/}
  V1_01_easy/mav0/cam1/{sensor.yaml,data.csv,data/}
  ...其余五条序列...
```

此处花括号仅用于说明目录，不是命令。将`EUROC_ROOT`替换成实际数据目录。代码不会自动下载数据；不需要训练模型。

```bash
python -m experiments.public_module.run_public_module --config experiments/contracts/public_module_config_v1.5.json --dataset-root EUROC_ROOT --sequence V1_01_easy --output outputs/euroc/V1_01_easy --profile formal
```

对上列全部六条序列分别运行，`--output`使用`outputs/euroc/序列名`。正式复现不要添加`--max-pairs`，也不要使用`tiny`替代正式规模。先做连通测试时可选择`--profile tiny --max-pairs 1`并使用单独输出目录，但该结果不能用于汇总证据。

六序列全部完成并检查各自`summary.json`后，在同一结果目录上运行两个范围的汇总：

```bash
python -m experiments.public_module.aggregate_public_module --config experiments/contracts/public_module_config_v1.5.json --input-root outputs/euroc --output outputs/euroc-six-review
python -m experiments.public_module.aggregate_public_module_v1_6 --base-config experiments/contracts/public_module_config_v1.5.json --scope experiments/contracts/public_module_scope_v1.6.json --input-root outputs/euroc --output outputs/euroc-five-review
```

退出码0只说明汇总程序正常完成；科学状态须读取`gate_review.json`。对新运行不要预设必须复现有利结果；若软件或数据差异改变样本资格，应保留差异并检查来源，不能静默变更范围。

汇总前会核对每条序列的`summary.json`、结果哈希、配置哈希、候选帧对数、方法组完整性、重复行、共同有效状态、有限指标及CE95/NLL的代数关系。零有效样本时指标和区间保持不可估计（JSON中为`null`），不输出NaN，也不通过删除该序列来计算较小范围的结果。生成图表前还会核对审查记录与统计表的哈希。
