# -*- coding: utf-8 -*-
"""
一键端到端运行与验收脚本
依次执行：
1. 数据分析与可视化 (step1_data_analysis.py)
2. 特征工程处理与特征选择 (step2_feature_engineering.py)
3. 24个个体多步预测模型训练与评估 (step3_individual_models.py)
4. 16个集成学习预测模型构建 (step4_ensemble_models.py)
5. 综合性能评价与可视化对比 (step5_evaluation_analysis.py)
6. 课程设计实践报告生成 (step6_generate_report.py)
7. 课程思政报告生成 (step7_generate_sizheng.py)
"""

import subprocess
import sys
import time
from pathlib import Path

scripts = [
    ("1. 数据分析与可视化 (对齐验收子任务1-5)", "step1_data_analysis.py"),
    ("2. 特征工程与特征选择 (对齐验收子任务6-8)", "step2_feature_engineering.py"),
    ("3. 24个个体预测模型训练 (ANN/RF/LSTM)", "step3_individual_models.py"),
    ("4. 16个集成预测模型构建 (Average/Stacking)", "step4_ensemble_models.py"),
    ("5. 全面指标评估与对比制图", "step5_evaluation_analysis.py"),
    ("6. 生成第一次阶段验收报告 (.docx)", "step1_2_generate_phase1_report.py"),
    ("7. 生成期末完整实践报告 (.docx)", "step6_generate_report.py"),
    ("8. 生成课程思政报告 (.docx)", "step7_generate_sizheng.py"),
]

def main():
    print("==================================================================")
    print("杭州电子科技大学《机器学习课程实践》—— 径流多步预测端到端全流程")
    print("==================================================================")
    total_start = time.time()
    
    for name, script in scripts:
        print(f"\n>>> 正在执行: {name} ({script}) ...")
        t0 = time.time()
        res = subprocess.run([sys.executable, str(Path(__file__).resolve().parent / script)], check=True)
        print(f">>> {name} 执行完成，耗时: {time.time() - t0:.2f} 秒")
        
    print("\n==================================================================")
    print(f"🎉 全部 7 个模块执行圆满成功！总耗时: {time.time() - total_start:.2f} 秒")
    print("所有生成交付物均已就绪，位于当前目录及 results/ 文件夹下。")
    print("==================================================================")

if __name__ == '__main__':
    main()
