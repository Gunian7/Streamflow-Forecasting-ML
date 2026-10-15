from pathlib import Path
import hashlib
import json
import shutil

root = Path(__file__).resolve().parent.parent
moved, blocked = [], []
for src in sorted(root.iterdir()):
    if not src.is_file() or src.name.startswith('~$'):
        continue
    dest = None
    if src.suffix.lower() == '.docx' and src.name.startswith(('《机器学习课程实践》', '思政报告-')):
        category = '最新报告' if '紧凑设计版' in src.name or src.name.startswith('思政报告-') else '历史版本'
        dest = root / '本地资料' / category / src.name
    elif src.suffix.lower() in ['.doc', '.docx', '.pdf']:
        dest = root / '本地资料' / '课程资料' / src.name
    elif src.name == '时间截图.png':
        dest = root / '本地资料' / '截图' / src.name
    if dest is None:
        continue
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        raise FileExistsError(dest)
    digest = hashlib.sha256(src.read_bytes()).digest()
    try:
        shutil.move(str(src), str(dest))
    except PermissionError:
        blocked.append(src.name)
        continue
    assert hashlib.sha256(dest.read_bytes()).digest() == digest
    moved.append(str(dest.relative_to(root)))
print(json.dumps({'moved': moved, 'blocked': blocked}, ensure_ascii=False, indent=2))
print('ROOT FILES')
for entry in sorted(root.iterdir()):
    print(entry.name + ('/' if entry.is_dir() else ''))
