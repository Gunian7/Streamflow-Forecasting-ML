from pathlib import Path
import pandas as pd
from docx import Document
import step2_feature_engineering as fe
import local_translate_csv as tr
r=Path(__file__).resolve().parent.parent
clean,full=fe.build_features(True)
d=fe.split_and_scale(clean)
p=r/'results/processed_data/01047000_with_feature_values_独热编码版.csv'
fe.export_feature_values(pd.read_csv(fe.DATA_PATH),full,d,p)
targets=[p,r/'results/figures/fe_without_current_discharge/feature_selection_ranking.csv']
lines=['# CSV字段说明（四季独热编码版）','','四季采用春、夏、秋、冬四个0/1列，每行恰有一列为1。归一化仅使用有效训练样本拟合参数，历史不足保留空白。完整CSV保留当天径流观测，特征排名排除当天径流。','']
for path in targets:
    backup=path.with_name(path.stem+'_英文备份.csv')
    backup.write_bytes(path.read_bytes())
    headers,count=tr.translate(path)
    lines += ['## '+path.name,'',f'{count}行，{len(headers)}列。','','|中文字段|英文字段|含义|','|---|---|---|']
    lines += ['|'+tr.describe(c)[0]+'|'+c+'|'+tr.describe(c)[1]+'|' for c in headers]
(r/'CSV字段说明.md').write_text('\n'.join(lines),encoding='utf-8')
for source in (r/'本地资料/最新报告').glob('*.docx'):
    if '思政' in source.name: continue
    doc=Document(source)
    for para in doc.paragraphs:
        if '【四季编码 (Season)】' in para.text:
            para.text=para.text.replace('【四季编码 (Season)】：按气象学标准将月份划分为春季 (3-5月=1)、夏季 (6-8月=2)、秋季 (9-11月=3)、冬季 (12-2月=0)。','【四季独热编码】：春季、夏季、秋季、冬季各占一列，所属季节取1，其他取0，不使用0/1/2/3类别编码。')
        if '四季编码表示日期类别' in para.text:
            para.text='四季采用独热编码：春季（3至5月）、夏季（6至8月）、秋季（9至11月）、冬季（12、1、2月）分别对应四个0/1列，每行恰有一个1。不把季节表示为0/1/2/3，避免引入大小关系。月份与年内日序仍保留正弦、余弦周期编码。'
        if '59个候选、57个保留' in para.text:
            para.text=para.text.replace('59个候选、57个保留','62个候选、59个保留')
        if '本次共评估 59 个候选特征' in para.text:
            para.text=para.text.replace('59 个候选特征','62 个候选特征').replace('筛选 57 个特征','筛选 59 个特征')
    for table in doc.tables:
        if any(cell.text=='四季编码' for cell in table.rows[0].cells):
            for row in table.rows:
                for cell in row.cells:
                    if cell.text=='四季编码': cell.text='四季独热（春/夏/秋/冬）'
            for row in table.rows[1:]:
                month=int(float(row.cells[1].text))
                row.cells[2].text='/'.join(str(int(month in ms)) for ms in [[3,4,5],[6,7,8],[9,10,11],[12,1,2]])
    doc.add_heading('四季独热编码更新',level=2)
    doc.add_paragraph('本版季节特征为春、夏、秋、冬四个指示变量，所属季节为1，其余为0。更新后的数值CSV为1827行、196列；特征选择排除当天径流，共62个候选，保留59个。已有模型未重训，旧模型指标不代表本版特征效果。')
    for shape in doc.inline_shapes:
        rel=doc.part.rels[shape._inline.graphic.graphicData.pic.blipFill.blip.embed]
        # Replace the feature-selection figure by exact comparison with previous image is not possible; append updated figure explicitly.
    doc.add_picture(str(r/'results/figures/fe_without_current_discharge/feature_selection_comparison.png'),width=__import__('docx').shared.Cm(16))
    out=source.with_stem(source.stem+'-独热编码版')
    doc.save(out)
    print(out)
check=pd.read_csv(p)
cols=['春季独热编码','夏季独热编码','秋季独热编码','冬季独热编码']
assert (check[cols].sum(axis=1)==1).all()
assert '四季编码' not in check.columns
print('VERIFIED',check.shape)
