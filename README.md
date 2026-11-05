# 机器学习课程实践 - 第三阶段（基于LSTM和集成学习的径流预测）

> **注**：第一、二阶段详细说明见 `README_phase1_2.md`

## 📁 项目结构

```
机器学习课程实践/
├── 01047000.csv                           # 原始数据
├── code/                                  # 核心代码
│   ├── phase2_v5_*.py                    # 第二阶段（集成学习需要）
│   ├── enhanced_features.py              # 增强特征工程（94维）
│   ├── pytorch_lstm.py                   # PyTorch-LSTM实现
│   ├── pytorch_lstm_grid_search.py       # Grid Search
│   ├── retrain_pytorch_lstm_final.py     # LSTM训练
│   ├── retrain_6models_ensemble.py       # 6模型集成
│   └── phase3_final_report.py            # 验收报告生成
├── results/
│   ├── phase2_v5/                        # 第二阶段模型
│   └── phase3_enhanced/                  # 第三阶段结果
├── 本地资料/第三阶段/
│   └── 第三阶段验收报告_正确顺序版.docx
└── archive/                               # 归档的废弃文件（51个文件，5个文件夹）
```

## 🎯 第三阶段任务（7个子任务全部完成 ✅）

| 子任务 | 方法 | 第1天R² | 第1天RMSE | 状态 |
|--------|------|---------|-----------|------|
| 1 | PyTorch-LSTM Grid Search | - | - | ✅ 8个任务 |
| 2 | PyTorch-LSTM直接策略 | **0.8641** | 313.0 cfs | ✅ **最优** |
| 3 | PyTorch-LSTM多输出策略 | 0.6810 | 479.5 cfs | ✅ |
| 4 | 6模型简单平均-直接 | 0.6593 | 495.5 cfs | ✅ |
| 5 | 6模型Stacking-直接 | 0.8175 | 365.6 cfs | ✅ |
| 6 | 3模型简单平均-多输出 | 0.2242 | - | ✅ |
| 7 | 3模型Stacking-多输出 | 0.7489 | - | ✅ |

## 🚀 快速开始

### 1. Grid Search（可选，约7分钟）
```bash
python code/pytorch_lstm_grid_search.py
```

### 2. 训练LSTM（约8分钟）
```bash
python code/retrain_pytorch_lstm_final.py
```

### 3. 生成验收报告（2分钟）
```bash
python code/phase3_final_report.py
```

## 📝 工程规范

✅ Grid Search（PyTorch-LSTM）→ 训练（PyTorch-LSTM）→ 报告（PyTorch-LSTM）  
✅ 每天使用各自的最优参数（非统一参数）  
✅ 逻辑完全一致

## 👥 团队

陈位恺 2022011270

---
**最后更新**: 2026-11-05 | **验收时间**: 2026-11-05 13:30
