"""GitHub推送准备：清理个人信息和无用文件"""
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# 需要删除的文件夹（包含个人信息或无用）
REMOVE_DIRS = [
    '本地资料',                    # 包含个人报告和文档
    'archive',                     # 归档的废弃文件
    '__pycache__',                 # Python缓存
    '.pytest_cache',               # Pytest缓存
]

# 需要删除的文件
REMOVE_FILES = [
    'README_phase1_2.md',          # 旧版README
]

# 需要删除的结果文件夹（保留代码，删除大文件结果）
REMOVE_RESULT_DIRS = [
    'results/phase2_v5',           # 第二阶段模型文件（.joblib很大）
    'results/phase3_enhanced',     # 第三阶段模型文件
]

# 保留的关键结果CSV（小文件，可以保留）
KEEP_RESULT_FILES = [
    'results/phase3_enhanced/pytorch_lstm_grid_summary.csv',
    'results/phase3_enhanced/pytorch_lstm_metrics.csv',
    'results/phase3_enhanced/6模型简单平均_直接.csv',
    'results/phase3_enhanced/enhanced_ensemble_metrics.csv',
]


def clean_for_github():
    """清理项目准备GitHub推送"""
    
    print('='*70)
    print('清理项目准备GitHub推送')
    print('='*70)
    print()
    
    removed = []
    
    # 删除文件夹
    print('删除个人信息和无用文件夹...')
    for dir_name in REMOVE_DIRS:
        dir_path = ROOT / dir_name
        if dir_path.exists():
            shutil.rmtree(dir_path)
            removed.append(dir_name)
            print(f'  ✓ 删除 {dir_name}/')
    
    # 删除文件
    print()
    print('删除无用文件...')
    for file_name in REMOVE_FILES:
        file_path = ROOT / file_name
        if file_path.exists():
            file_path.unlink()
            removed.append(file_name)
            print(f'  ✓ 删除 {file_name}')
    
    # 保留部分CSV结果，删除模型文件
    print()
    print('清理结果文件夹（保留CSV，删除模型）...')
    
    # 先备份要保留的CSV
    kept_files = {}
    for csv_path in KEEP_RESULT_FILES:
        full_path = ROOT / csv_path
        if full_path.exists():
            import pandas as pd
            kept_files[csv_path] = pd.read_csv(full_path)
            print(f'  ✓ 备份 {csv_path}')
    
    # 删除结果文件夹
    for dir_name in REMOVE_RESULT_DIRS:
        dir_path = ROOT / dir_name
        if dir_path.exists():
            shutil.rmtree(dir_path)
            print(f'  ✓ 删除 {dir_name}/')
    
    # 恢复CSV文件
    for csv_path, df in kept_files.items():
        full_path = ROOT / csv_path
        full_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(full_path, index=False, encoding='utf-8-sig')
        print(f'  ✓ 恢复 {csv_path}')
    
    print()
    print('='*70)
    print('清理完成！')
    print('='*70)
    print()
    print(f'已删除: {len(removed)} 项')
    print()
    
    # 显示保留的文件
    print('保留的核心文件:')
    core_files = [
        'code/*.py (Python代码)',
        '01047000.csv (原始数据)',
        'README.md (项目说明)',
        'results/phase3_enhanced/*.csv (关键结果CSV)',
    ]
    for f in core_files:
        print(f'  ✓ {f}')
    
    print()
    print('下一步: 创建.gitignore和推送到GitHub')


if __name__ == '__main__':
    clean_for_github()
