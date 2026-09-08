# DUC-VIO

本仓库提供 DUC-K16 残差尺度校准方法的最小复现实验集合。当前实验包为 v1.0.1。

研究对象是优化后残差的尺度校准。公开包保留支撑的四类实验、必要对照和边界结果，不包含完整 VIO 系统、神经网络训练或轨迹精度评价。

## 实验与证据

| 实验 | 冻结规模 | 用途 |
|---|---|---|
| 线性块校准与随机探针选择 | 27个设置，各500个问题，共13,500个 | 检查校准、留一因子近似以及K=4、8、16的近似误差 |
| 非线性重投影 | 200个问题，每个抽取8个因子做留一检查 | 检查非线性条件下的校准点估计与不确定性 |
| 顺序尺度跟踪 | 8个设置，各200次，共1,600次 | 检查精确标量块迹校正下的尺度响应和边界命中 |
| EuRoC独立留出评价 | 六条Vicon Room序列，两种拟合集大小N=16/48，五种方法 | 评价真实图像条件下的局部校准及校正增量时间 |

前三类使用按冻结模型生成的受控合成数据，属于已执行的数值实验，不是代替真实实验的占位数据。EuRoC使用真实公开数据；仓库包含逐帧对指标，不分发原始图像。

主要公开结果为N=16、五条有效样本充足序列上的评价，共894个共同有效帧对。N=48、V2_03样本不足结果和置换对照全部保留。当前证据不支持完整系统性能或唯一作用机制的结论。详见[实验范围](docs/EXPERIMENTS.md)。

## 快速检查：无需数据集或训练

在仓库根目录使用Python 3.10。建议独立环境；已有兼容环境可直接使用。

```bash
python -m pip install -r requirements.txt
python -m pytest -q
python scripts/run_e0.py --profile smoke --output outputs/e0-smoke
python scripts/reproduce.py --output outputs/paper-recheck --plots
```

输出目录必须尚不存在；重跑时换用新目录。`smoke`每类只运行一个缩小问题，不能用于正式结论。`reproduce.py`不运行新科学实验：它从六条序列的逐帧对结果重算EuRoC汇总，核对归档结果，并用已核验图源重绘三幅定量图。原六序列门槛未通过的状态也必须匹配，不能改成通过。

运行后，`outputs/paper-recheck/`包含重算表格、SVG/PDF/PNG图和`reproduction_report.json`。默认不上传生成的图像和临时结果。图的证据范围见[图形说明](docs/FIGURES.md)。

## 文件导航

| 路径 | 内容 |
|---|---|
| `experiments/e0/e0formal/` | 六个原始数值内核与两个公开离线入口适配文件 |
| `experiments/e0/config/e0_v1_4.json` | 冻结数值实验配置 |
| `experiments/public_module/` | EuRoC加载、特征筛选、求解、评价、结果校验、汇总和绘图代码 |
| `experiments/artifacts/` | 六序列逐帧对指标、全范围与收窄范围汇总 |

完整复现命令、数据目录及环境差异见[实验范围](docs/EXPERIMENTS.md)；可复现程度、时间指标限制和发布检查见[来源说明](docs/PROVENANCE.md)。

## 数据和使用范围

EuRoC数据请从[ETH官方页面](https://projects.asl.ethz.ch/datasets/euroc-mav/)及其链接的[数据存档](https://doi.org/10.3929/ethz-b-000690084)获取，并遵守数据提供方的使用条件。数据论文：Burri et al., *The EuRoC micro aerial vehicle datasets*, IJRR, 2016, [doi:10.1177/0278364915620033](https://doi.org/10.1177/0278364915620033)。

本次仅发布实验材料，不代填作者、单位、资助或通信信息，也不代替权利人指定开源许可。仓库尚未附加软件许可证；公开可访问不等于授予任意再分发或商业使用权限。
