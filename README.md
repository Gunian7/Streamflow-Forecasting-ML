# Streamflow-Forecasting-ML (基于机器学习的流域径流多步预测)

本项目为基于气象驱动因子（Daymet）与历史水文观测（USGS 01047000 流域）的日尺度径流多步预报系统。系统实现了从探索性数据分析、物理机制特征工程、多模型时序网格搜索建模到异构集成学习的全流程闭环。

---

## 目录结构说明

```text
Streamflow-Forecasting-ML/
│
├── 01047000.csv                               # 原始水文气象数据集 (USGS NWIS + NASA Daymet 逐日序列)
├── .gitignore                                 # Git 忽略配置
├── README.md                                  # 项目开源公开说明文档
│
├── run_pipeline.py                            # 一键全自动端到端运行总脚本 (串联 step1 ~ step5 及报告生成)
├── step1_data_analysis.py                     # 一、探索性数据分析与分布形态可视化 (时间序列、直方图、箱型图、皮尔逊热力图)
├── step2_feature_engineering.py               # 二、物理机制特征工程 (四季/循环编码、滞后阶数、滑动统计窗、特征选择)
├── step3_individual_models.py                 # 三、24个个体多步预测模型（ANN、RF、LSTM × 8）+ 时序网格搜索交叉验证
├── step4_ensemble_models.py                   # 四、16个集成学习预测模型（Simple Average、Stacking 堆叠泛化 × 8）
├── step5_evaluation_analysis.py               # 五、综合多指标评价体系（RMSE, MAE, R², NSE）与水文过程线拟合制图
├── step1_2_generate_phase1_report.py          # 六、阶段数据分析与特征工程成果文档生成脚本
├── step6_generate_report.py                   # 七、完整实践分析报告自动化生成脚本
├── step7_generate_sizheng.py                  # 八、配套工程伦理与智慧水利报告生成脚本
│
└── results/                                   # 试验与评测产物输出目录
    ├── figures/                               # 自动生成的高清学术可视化图表 (300 DPI)
    │   ├── eda/                               # 探索性数据分析图表 (时间序列全景、直方图KDE、8列独立箱型图、滞后相关热力图)
    │   ├── fe/                                # 特征工程图表 (Min-Max/Z-Score分布对比、特征重要度与互信息对比)
    │   └── evaluation/                        # 评估图表 (预见期误差衰减曲线、策略对比柱状图、春汛水文过程线拟合)
    ├── metrics/                               # 详细量化指标汇总表格 (CSV，含每个步长各模型精度)
    ├── predictions/                           # 预测矩阵与统一对齐数据包 (.pkl)
    └── processed_data/                        # 清洗与特征工程后严格时序切分的数据包 (.pkl)
```

---

## 核心算法架构与预测策略

本工程针对未来 1~7 天径流量预测，设计并对比了两种多步预报策略：
1. **多步直接预测策略 (Direct Multi-step Forecasting)**：为未来第 1 天至第 7 天分别训练独立优化的回归预测器（共 $3 \times 7 = 21$ 个基模型），避免误差沿时间步自回归递归累积。
2. **多输出预测策略 (Multi-output Forecasting)**：构建联合输出向量预测器（$3 \times 1 = 3$ 个基模型），利用模型的隐层表征同时捕获多步长之间的协同演化动态。

### 涵盖的模型族系：
- **人工神经网络 (ANN, MLPRegressor)**：非线性映射网络，时序网格搜索最佳隐层神经元配置。
- **随机森林 (Random Forest Regressor)**：非参数集成树模型，对极值波动具备良好鲁棒性。
- **长短期记忆网络 (LSTM, PyTorch)**：深度时序递归神经网络，提取多尺度时间依赖记忆。
- **集成学习 (Ensemble Learning)**：
  - **简单平均集成 (Simple Average)**：等权融合各异构模型预测输出，降低模型方差。
  - **Stacking 堆叠泛化 (Ridge Meta-Learner)**：采用带 L2 正则化的岭回归作为元学习器，拟合基模型在各步长上的动态置信度权重。

---

## 运行与复现指南

### 1. 环境准备
推荐使用 Python 3.10+ 环境（建议通过 `uv` 或 `venv` 构建独立虚拟环境）：
```bash
# 安装核心科学计算、深度学习及文档依赖
pip install pandas numpy scipy scikit-learn matplotlib seaborn torch python-docx
```

### 2. 一键执行完整工作流
在项目根目录下直接运行流水线脚本：
```bash
python run_pipeline.py
```
脚本将按顺序自动执行：
1. `step1_data_analysis.py`：生成数据全景分布与相关性分析图表并输出描述统计。
2. `step2_feature_engineering.py`：构建 60 维时序物理特征空间，利用皮尔逊与互信息法筛选 58 个最优特征子集。
3. `step3_individual_models.py`：网格搜索超参数并训练 24 个基学习器。
4. `step4_ensemble_models.py`：构建 16 个直接预测与多输出集成学习模型。
5. `step5_evaluation_analysis.py`：全样本对齐并计算 RMSE、MAE、$R^2$ 及水文学纳什效率系数 (NSE)，生成多维度评估图表。
6. `step1_2_generate_phase1_report.py` / `step6_generate_report.py` / `step7_generate_sizheng.py`：自动装配并输出对应分析报告文档。

---

## 核心量化评估结论

在 2004 年未参与训练的真实测试样本上，主要评估结论如下：

| 算法模型 | 预测策略 | Lead 1 RMSE (cfs) | Lead 1 MAE (cfs) | Lead 1 NSE (纳什系数) | 7天平均 NSE |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **简单平均集成** | **多步直接** | **596.96** | **226.41** | **0.5000** | **0.1702** |
| **简单平均集成** | **多输出** | **600.65** | **230.94** | **0.4938** | **0.1862** |
| LSTM | 多步直接 | 616.65 | 271.84 | 0.4665 | 0.1321 |
| 随机森林 | 多输出 | 627.12 | 258.65 | 0.4483 | -0.0150 |
| ANN | 多步直接 | 658.08 | 318.38 | 0.3924 | 0.0869 |
| Stacking集成 | 多步直接 | 687.82 | 265.91 | 0.3363 | -0.5823 |

- **第 1 天预测表现**：简单平均集成模型拔得头筹（$\text{NSE} = 0.5000$），显著优于任何单一基模型，有效缓解了单一模型在春季融雪特大洪峰期间的削峰滞后问题。
- **预见期衰减特性**：随着预测步长由 1 天延展至 7 天，因无未来真实降水驱动输入，各模型精度呈物理规律性衰减，量化揭示了气象时滞对产汇流可预报性的物理约束。
