import json
from pathlib import Path


def member_fields():
    path = Path(__file__).resolve().parent.parent / 'local_report_members.json'
    if not path.exists():
        return '（学生学号）', '（学生姓名）'
    members = json.loads(path.read_text(encoding='utf-8'))
    return ('\n'.join(str(m['student_id']) for m in members),
            '\n'.join(m['name'] for m in members))
