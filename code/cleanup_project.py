"""整理项目，移除不必要的文件"""
import os
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# 需要保留的核心文件
KEEP_FILES = {
    # 核心数据处理
    'code/phase2_v5_data.py',          # 第二阶段数据（子任务6-7使用）
    'code/enhanced_features.py',       # 增强特征（PyTorch-LSTM使用）
    
    # 第三阶段核心
    'code/phase3_lstm.py',             # PseudoLSTM实现
    'code/pytorch_lstm.py',            # PyTorch-LSTM实现
    'code/pytorch_lstm_grid_search.py',# PyTorch-LSTM Grid Search
    'code/retrain_pytorch_lstm_final.py', # PyTorch-LSTM训练
    
    # 集成学习
    'code/retrain_6models_ensemble.py', # 6模型集成（子任务4-5）
    'code/retrain_multi_3models.py',    # 3模型集成（子任务6-7）
    
    # 文档生成
    'code/phase3_final_report.py',     # 最终验收报告生成
    
    # 数据文件
    '01047000.csv',                     # 原始数据
}

# 需要保留的结果文件夹
KEEP_RESULT_DIRS = {
    'results/phase3_enhanced',         # PyTorch-LSTM结果
    'results/phase2_v5',               # 第二阶段模型（集成学习需要）
}

# 可以删除的废弃文件（诊断、实验、旧版本）
REMOVE_PATTERNS = [
    'code/diag_*.py',                  # 诊断脚本
    'code/phase2_v3_*.py',             # 第二阶段v3（已废弃）
    'code/phase2_v4_*.py',             # 第二阶段v4（已废弃）
    'code/local_*.py',                 # 本地临时脚本
    'code/build_fixed_dataset.py',     # 数据修复（已完成）
    'code/fix_chart_font.py',          # 图表修复（已完成）
    'code/grid_search_all_days.py',    # PseudoLSTM Grid Search（已废弃）
    'code/retrain_lstm_with_best_params.py', # PseudoLSTM训练（已废弃）
    'code/retrain_multi_ensemble.py',  # 旧版集成脚本
    'code/retrain_direct_*.py',        # 旧版训练脚本
    'code/phase2_*.py',                # 第二阶段其他脚本
    'code/phase3_ensemble.py',         # 旧版集成脚本
    'code/enhanced_ensemble.py',       # 旧版增强集成
    'code/generate_enhanced_report.py',# 旧版报告生成
    'code/train_pytorch_lstm.py',      # 旧版训练脚本
    '01047000_fixed.csv',              # 修复后的数据（已合并到原始文件）
]

# 可以删除的结果文件夹
REMOVE_RESULT_DIRS = [
    'results/phase2_presentation',
    'results/phase2_v3',
    'results/phase2_v4',
    'results/phase3_lstm',             # PseudoLSTM结果（已废弃）
    'results/phase3_ensemble',         # 旧版集成结果
    'results/phase3_figures',          # 旧版图表
]


def scan_files():
    """扫描可删除的文件"""
    to_remove = []
    
    # 扫描代码文件
    code_dir = ROOT / 'code'
    if code_dir.exists():
        for pattern in REMOVE_PATTERNS:
            if '*' in pattern:
                import glob
                for f in glob.glob(str(ROOT / pattern)):
                    if f not in [str(ROOT / k) for k in KEEP_FILES]:
                        to_remove.append(f)
            else:
                f = ROOT / pattern
                if f.exists() and str(f.relative_to(ROOT)) not in KEEP_FILES:
                    to_remove.append(str(f))
    
    return to_remove


def main():
    print('='*70)
    print('项目整理 - 识别可删除文件')
    print('='*70)
    print()
    
    to_remove = scan_files()
    
    if not to_remove:
        print('✓ 没有找到需要删除的文件')
        return
    
    print(f'找到 {len(to_remove)} 个可删除的文件:')
    print()
    
    # 按类别分组
    diag_files = [f for f in to_remove if 'diag_' in f]
    phase2_files = [f for f in to_remove if 'phase2_' in f]
    local_files = [f for f in to_remove if 'local_' in f]
    other_files = [f for f in to_remove if f not in diag_files + phase2_files + local_files]
    
    if diag_files:
        print(f'诊断脚本 ({len(diag_files)}个):')
        for f in diag_files[:5]:
            print(f'  - {Path(f).name}')
        if len(diag_files) > 5:
            print(f'  ... 还有 {len(diag_files)-5} 个')
        print()
    
    if phase2_files:
        print(f'第二阶段旧版本 ({len(phase2_files)}个):')
        for f in phase2_files[:5]:
            print(f'  - {Path(f).name}')
        if len(phase2_files) > 5:
            print(f'  ... 还有 {len(phase2_files)-5} 个')
        print()
    
    if local_files:
        print(f'本地临时脚本 ({len(local_files)}个):')
        for f in local_files[:5]:
            print(f'  - {Path(f).name}')
        if len(local_files) > 5:
            print(f'  ... 还有 {len(local_files)-5} 个')
        print()
    
    if other_files:
        print(f'其他废弃文件 ({len(other_files)}个):')
        for f in other_files:
            print(f'  - {Path(f).name}')
        print()
    
    print('='*70)
    print('结果文件夹整理建议:')
    print('='*70)
    print()
    print('可以删除:')
    for d in REMOVE_RESULT_DIRS:
        if (ROOT / d).exists():
            print(f'  - {d}')
    print()
    print('需要保留:')
    for d in KEEP_RESULT_DIRS:
        if (ROOT / d).exists():
            print(f'  - {d}')
    print()
    
    print('='*70)
    print('保留的核心文件:')
    print('='*70)
    print()
    for f in sorted(KEEP_FILES):
        if (ROOT / f).exists():
            print(f'  ✓ {f}')
    print()
    
    # 创建备份文件夹
    archive = ROOT / 'archive'
    print(f'建议: 将废弃文件移动到 {archive} 而不是直接删除')
    print('      这样可以在需要时恢复')


if __name__ == '__main__':
    main()
