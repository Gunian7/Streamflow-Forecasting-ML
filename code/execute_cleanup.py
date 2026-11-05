"""执行项目清理，将废弃文件移动到archive"""
import os
import shutil
from pathlib import Path
import glob

ROOT = Path(__file__).resolve().parent.parent
ARCHIVE = ROOT / 'archive'

# 需要保留的核心文件
KEEP_FILES = {
    'code/phase2_v5_data.py',
    'code/phase2_v5_ensemble.py',
    'code/phase2_v5_baselines.py',
    'code/phase2_v5_experiment.py',
    'code/enhanced_features.py',
    'code/phase3_lstm.py',
    'code/pytorch_lstm.py',
    'code/pytorch_lstm_grid_search.py',
    'code/retrain_pytorch_lstm_final.py',
    'code/retrain_6models_ensemble.py',
    'code/retrain_multi_3models.py',
    'code/phase3_final_report.py',
    '01047000.csv',
}

# 删除模式
REMOVE_PATTERNS = [
    'code/diag_*.py',
    'code/phase2_v3_*.py',
    'code/phase2_v4_*.py',
    'code/local_*.py',
    'code/build_fixed_dataset.py',
    'code/fix_chart_font.py',
    'code/grid_search_all_days.py',
    'code/retrain_lstm_with_best_params.py',
    'code/retrain_multi_ensemble.py',
    'code/phase2_diagnostics.py',
    'code/phase2_experiment.py',
    'code/phase2_presentation.py',
    'code/phase2_report.py',
    'code/phase3_ensemble.py',
    'code/enhanced_ensemble.py',
    'code/generate_enhanced_report.py',
    'code/train_pytorch_lstm.py',
    'code/train_boosting.py',
    '01047000_fixed.csv',
]

REMOVE_RESULT_DIRS = [
    'results/phase2_v3',
    'results/phase2_v4',
    'results/phase3_lstm',
    'results/phase3_ensemble',
    'results/phase3_figures',
]


def move_to_archive(src, archive_base):
    """移动文件到archive，保持目录结构"""
    rel_path = src.relative_to(ROOT)
    dst = archive_base / rel_path
    dst.parent.mkdir(parents=True, exist_ok=True)
    
    if src.is_dir():
        if dst.exists():
            shutil.rmtree(dst)
        shutil.move(str(src), str(dst))
    else:
        if dst.exists():
            dst.unlink()
        shutil.move(str(src), str(dst))
    
    return dst


def main():
    print('='*70)
    print('执行项目清理')
    print('='*70)
    print()
    
    # 创建archive目录
    ARCHIVE.mkdir(exist_ok=True)
    print(f'✓ 创建归档目录: {ARCHIVE}')
    print()
    
    moved_files = []
    moved_dirs = []
    
    # 移动废弃文件
    print('移动废弃代码文件...')
    for pattern in REMOVE_PATTERNS:
        for src in glob.glob(str(ROOT / pattern)):
            src_path = Path(src)
            rel = src_path.relative_to(ROOT)
            
            if str(rel) in KEEP_FILES:
                continue
            
            if src_path.exists():
                dst = move_to_archive(src_path, ARCHIVE)
                moved_files.append(str(rel))
                print(f'  ✓ {rel}')
    
    print()
    print(f'已移动 {len(moved_files)} 个文件')
    print()
    
    # 移动废弃结果文件夹
    print('移动废弃结果文件夹...')
    for dir_path in REMOVE_RESULT_DIRS:
        src = ROOT / dir_path
        if src.exists():
            dst = move_to_archive(src, ARCHIVE)
            moved_dirs.append(dir_path)
            print(f'  ✓ {dir_path}')
    
    print()
    print(f'已移动 {len(moved_dirs)} 个文件夹')
    print()
    
    # 显示保留的文件
    print('='*70)
    print('保留的核心文件:')
    print('='*70)
    print()
    
    kept_files = []
    for f in sorted(KEEP_FILES):
        if (ROOT / f).exists():
            kept_files.append(f)
            print(f'  ✓ {f}')
    
    print()
    print(f'保留 {len(kept_files)} 个核心文件')
    print()
    
    # 创建README
    readme = ARCHIVE / 'README.md'
    with open(readme, 'w', encoding='utf-8') as f:
        f.write('# 归档文件\n\n')
        f.write('这些文件是项目整理时移动到此处的废弃文件，包括：\n\n')
        f.write('## 废弃代码\n\n')
        for file in moved_files:
            f.write(f'- {file}\n')
        f.write('\n## 废弃结果文件夹\n\n')
        for dir in moved_dirs:
            f.write(f'- {dir}/\n')
        f.write('\n如需恢复，可以手动移回原位置。\n')
    
    print(f'✓ 创建归档说明: {readme}')
    print()
    print('='*70)
    print('清理完成！')
    print('='*70)
    print()
    print('归档位置:', ARCHIVE)
    print('如需恢复，可以从archive文件夹中取回')


if __name__ == '__main__':
    main()
