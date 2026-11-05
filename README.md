# 站点 01047000 逐日径流多步预测

本项目为《机器学习课程实践》课程作业。以 USGS 站点 01047000 的逐日水文气象观测序列为对象，完成数据探索、特征工程、单模型建模、集成学习与 LSTM 多步预测的完整流程，并生成第一、二、三次阶段验收报告。报告中出现的每一项数值均由脚本在运行时读取 `results/` 下的结果文件后格式化输出，不作人工改写。

## 一、数据说明

工作目录下并存两个数据文件，列结构完全一致（9 列：`Date`、`Discharge`、`Dayl`、`Prcp`、`Srad`、`Swe`、`Tmax`、`Tmin`、`Vp`），均为 1827 行，覆盖 2000-01-01 至 2004-12-31 的连续日序列；径流单位为 cfs，取自 USGS 实测值。

| 文件 | 性质 | 用途 | 积雪水当量 |
|---|---|---|---|
| `01047000.csv` | 原始表，保持课程下发时的原样，不作改动 | 对账与留档 | `Swe` 全序列恒为 0，且气象列与 Daymet 官方日值整体相差一天 |
| `01047000_fixed.csv` | 修正表，由 `code/build_fixed_dataset.py` 生成 | 建模唯一数据源 | `Swe` 按 Daymet 日值按 `year`+`yday` 对齐后回落，866 天非零 |

修正表相对原始表只做两件事：把 Daymet 的气象变量按官方日历日重新对齐，以及恢复真实的积雪水当量；径流列沿用 USGS 实测值不变。两表的行数与日期范围一致，因此可以直接替换。

## 二、目录结构

```text
机器学习课程实践/
├── 01047000.csv                      # 原始数据表（只留档，不参与建模）
├── 01047000_fixed.csv                # 修正数据表（建模唯一数据源）
├── code/                             # 26 个保留脚本，见下
├── tests/                            # 9 个测试模块，pytest 采集 39 项
├── results/
│   ├── phase2_v5/                    # 第二阶段最终结果（28 行精度指标 + 16 份网格搜索 + 逐日预测 + 3 张图）
│   ├── phase3_enhanced/              # 第三阶段最终结果（LSTM 精度、四张集成表、权重、94 维特征清单、4 张图）
│   ├── phase3_lstm/                  # 第三阶段 LSTM 中间产物（15 个文件）
│   ├── processed_data/               # 特征工程输出表（196 列，含中英双列名）
│   ├── figures/eda|fe_*/             # 第一阶段图表与特征排名
│   ├── diagnostics/|probe/|phase2_v3/# 数据核对、逐期探针与 v3 试验留档
│   └── final_summary.md              # 由结果文件汇总的总表
├── archive/                          # 只移动不删除的历史内容
│   ├── code_legacy/                  # 29 个历史脚本（重训、旧报告生成器、整理工具）
│   ├── results_legacy/               # 旧日志与旧结果（logs/ 下 5 个训练日志）
│   ├── results_phase2_v5_raw/        # 未修正数据版的第二阶段结果（只读对照）
│   ├── results_phase3_enhanced_raw/  # 未修正数据版的第三阶段结果（只读对照）
│   ├── results_phase3_lstm_raw/      # 未修正数据版的 LSTM 结果（只读对照）
│   └── _scripts/                     # 一次性核对脚本
├── 本地资料/                          # 三份验收报告副本（第一阶段/第二阶段/第三阶段）与历史版本
├── 文件结构说明.md                    # 逐脚本用途表与归档位置
├── CSV字段说明.md                     # 特征表逐字段释义
└── local_report_members.json         # 封面姓名/学号/指导教师，已被 .gitignore 忽略
```

## 三、阶段任务完成情况

### 3.1 第一阶段：数据分析与特征工程（8 个子任务，全部完成）

| 子任务 | 内容 | 产物 |
|---|---|---|
| 1 | 数据字段英文名与中文释义 | 表 1-1（9 个字段） |
| 2 | 时间序列可视化 | `results/figures/eda/` 过程线 |
| 3 | 直方图与分布形态 | 分布统计表与直方图 |
| 4 | 箱型图离群检测 | 离群记录统计 |
| 5 | 皮尔逊相关系数热力图 | 相关矩阵与滞后相关表 |
| 6 | 时间序列特征提取 | 196 列特征表 |
| 7 | 特征归一化（MinMax / Z-score） | 归一化列 |
| 8 | 特征选择（互信息 + 相关系数双指标） | 62 个候选中入选 59 个，排除当天径流 |

### 3.2 第二阶段：神经网络与随机森林（全部完成）

基础特征 71 维，训练段 2000—2003 年、测试段 2004 年（359 行），选参只用 `TimeSeriesSplit(n_splits=3, gap=7)`，测试集仅做一次最终评估。第 1 天（预见期 1 天）结果：

| 模型 | 策略 | RMSE（cfs） | MAE（cfs） | R² |
|---|---|---|---|---|
| ANN | 直接 | 443.1677 | 177.2011 | 0.7230 |
| ANN | 多输出 | 425.0276 | 182.8553 | 0.7452 |
| RF | 直接 | 539.9305 | 152.6497 | 0.5888 |
| RF | 多输出 | 605.5606 | 194.0161 | 0.4828 |

等权融合（权重固定，不在测试集上调参）：ANN 两策略等权 R²=0.7878，四模型等权 R²=0.7146，ANN 多输出 + RF 多输出 R²=0.6940。精度随预见期单调衰减，ANN 直接策略在第 5 天起 R² 转负。

### 3.3 第三阶段：LSTM 与集成学习（7 个子任务，全部完成）

增强特征 94 维，训练 1434 行、测试 353 行。第 1 天结果：

| 子任务 | 方法 | RMSE（cfs） | MAE（cfs） | R² |
|---|---|---|---|---|
| 1 | PyTorch-LSTM 网格搜索（8 个任务） | — | — | 交叉验证选定参数 |
| 2 | PyTorch-LSTM 直接策略 | 332.8969 | 133.7564 | 0.8463 |
| 3 | PyTorch-LSTM 多输出策略 | 377.8962 | 193.9315 | 0.8019 |
| 4 | 6 模型简单平均（直接） | 360.5866 | 119.7659 | 0.8196 |
| 5 | 6 模型 Stacking（直接） | 388.5545 | 129.0447 | 0.7905 |
| 6 | 3 模型简单平均（多输出） | 409.6773 | 151.5224 | 0.7672 |
| 7 | 3 模型 Stacking（多输出） | 486.7773 | 220.0371 | 0.6713 |

## 四、复现步骤

全部命令在项目根目录执行（Windows 环境，虚拟环境位于 `.venv/`）：

```bash
.venv/Scripts/python.exe code/build_fixed_dataset.py      # 生成 01047000_fixed.csv
.venv/Scripts/python.exe code/step1_data_analysis.py      # 第一阶段：探索性分析
.venv/Scripts/python.exe code/step2_feature_engineering.py # 第一阶段：特征表（196 列）
.venv/Scripts/python.exe code/phase2_v5_experiment.py     # 第二阶段：网格搜索与训练
.venv/Scripts/python.exe code/phase2_v5_figs.py           # 第二阶段：三张结果图
.venv/Scripts/python.exe code/phase3_run_all.py           # 第三阶段：7 个 stage 全量运行
.venv/Scripts/python.exe code/phase2_report.py            # 生成第二次阶段验收报告
.venv/Scripts/python.exe code/phase3_report.py            # 生成第三次阶段验收报告
.venv/Scripts/python.exe code/phase1_report.py            # 生成第一次阶段验收报告
.venv/Scripts/python.exe code/verify_all_results.py       # 结果自洽性校验
.venv/Scripts/python.exe code/make_summary.py             # 汇总 results/final_summary.md
.venv/Scripts/python.exe -m pytest tests -q               # 39 项测试
```

报告生成器会把成品写入 `本地资料/第N阶段/` 并同步一份到项目根，两处内容逐字节一致。封面姓名、学号与指导教师不在脚本内，统一从 `local_report_members.json` 读取（该文件已被 `.gitignore` 忽略；文件缺失时退回占位符并打印警告，不中断生成）。

复现耗时（同一台机器实测，仅供参考）：

| 环节 | 耗时 |
|---|---|
| `build_fixed_dataset.py` | 0.8 s |
| 第二阶段全量训练（`phase2_v5_experiment.py`） | 704 s（日志记录） |
| 第三阶段（`phase3_run_all.py`，7 个 stage） | 约 1933 s（stage1 1040.6 s、stage2 707.7 s、stage3 9.3 s、stage4 36.3 s、stage5 137.5 s、stage6 1.6 s、stage7 0.1 s） |
| 三份报告生成（`phase1_report.py`、`phase2_report.py`、`phase3_report.py`） | 1.6 s、1.5 s、1.6 s |
| `verify_all_results.py` 与 `make_summary.py` | 5.9 s、0.8 s |
| `pytest tests` | 3.4 s |

因此完整重跑一次约需 45 分钟，其中训练占绝大部分；只重生成报告不超过 10 s。

## 五、环境依赖与版本

- Python 3.14.7（`.venv`）
- numpy 2.5.4、pandas 3.1.1、scipy 1.18.1
- scikit-learn 1.9.1、joblib 1.6.0
- matplotlib 3.11.2、seaborn 0.13.2
- torch 2.15.0
- xgboost 3.4.1、lightgbm 4.7.0
- python-docx 1.2.0（报告生成）
- pytest 9.2.0

## 六、已知限制

1. **积雪只有 8 个派生特征**：`Swe`、`Swe_melt1`、`Swe_melt_3`、`Swe_melt_7`、`Swe_roll_mean_7`、`PDD`、`PDD_3`、`Snow_cover`。若数据源为原始表（`Swe` 恒为 0），这些列全部因零方差被剔除，`tests/test_phase2_v5_data.py` 的积雪断言此时会以 skip 形式说明，而不是假装通过。
2. **测试集口径不同**：第二阶段基础特征 71 维，测试段 359 行；第三阶段增强特征 94 维含 7 日滚动统计，首尾缺失行被丢弃，测试段 353 行。两阶段的单日指标不能直接相减比较。
3. **LSTM 的序列长度为 1**：每个样本只是起报时刻的单步特征向量，未构造多日滑动窗口，因此模型并未真正利用长程时序依赖，`LSTM` 与时序网络之外的模型差距有限。
4. **Stacking 劣于简单平均**：直接策略 0.7905 vs 0.8196，多输出策略 0.6713 vs 0.7672。元学习器（标准化 + TimeSeriesSplit 选 alpha 的 Ridge）只在训练段末尾 287 行留出块上拟合，样本量小且基学习器高度同质，权重估计方差大，等权平均反而更稳。
5. 预见期越长精度越差，第 4 天以后多数方案 R² 接近 0 或转负，多步外推能力有限。

## 七、配套文档

- `文件结构说明.md`：逐脚本用途表与归档位置
- `CSV字段说明.md`：特征表逐字段释义
- `results/final_summary.md`：由结果文件自动汇总的总表
