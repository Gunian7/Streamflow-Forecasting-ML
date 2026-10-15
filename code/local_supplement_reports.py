from pathlib import Path
import pandas as pd
from docx import Document
from docx.shared import Pt

ROOT = Path(__file__).resolve().parent.parent
CSV = ROOT / 'results/processed_data/01047000_with_feature_values_独热编码版.csv'
TITLE = '特征数值表的生成方法与结果解释'

def supplement(path):
    df = pd.read_csv(CSV)
    doc = Document(path)
    if any(p.text == TITLE for p in doc.paragraphs):
        raise ValueError('Supplement already exists')
    start = len(doc.element.body)
    doc.add_heading(TITLE, level=2)
    sections = [
        ('1. CSV 的组织与日期对齐',
         f'数值表 {CSV.name} 共 {len(df)} 行、{len(df.columns)} 列。每行对应一个观测日期，原始日期、径流及气象列保留在左侧，右侧依次追加日期编码、滞后值、窗口统计和归一化结果。程序按日期对齐后横向拼接，不增加或打乱原始观测行。原始 01047000.csv 不覆盖，完整数值与字段说明分别见该 CSV 和 CSV字段说明.md。'),
        ('2. 四季与月份周期编码',
         '四季编码表示日期类别：冬季（12、1、2月）为0，春季（3至5月）为1，夏季（6至8月）为2，秋季（9至11月）为3。数字不是水量大小，也不代表季节之间存在等距关系。现有程序使用整数编码；需要避免大小关系时可考虑独热编码。月份正弦和余弦分别为 sin(2π×月份/12) 与 cos(2π×月份/12)，二者配对表达圆周位置，使12月与1月保持相邻。年内日序采用相同思路，分母为365.25。它们提供季节位置，无物理单位，不是新的实测气象变量。'),
        ('3. 滞后特征的数值含义',
         '对于日期t，滞后k天的值等于t-k天的观测值。程序对径流、降水、最高/最低气温、短波辐射和水汽压分别提取1、2、3、7天滞后值，使用 shift(k) 实现。当前输入为连续逐日数据，因此移动k行对应k天。历史值可反映流域前期水量、降水和气象状态，为预测未来第1天径流提供历史信息。历史不足的单元格为空，不意味着观测值为零。'),
        ('4. 窗口统计的数值含义',
         '先 shift(1)，再 rolling(w)，在日期t聚合t-w至t-1的历史观测，不包含当天。对径流、降水、最高气温，分别计算过去3天和7天的均值、标准差、最大值和最小值。均值概括近期水平；标准差概括波动；最大值和最小值概括近期上下界。标准差采用样本标准差（ddof=1）。各统计量保留原变量单位；窗口不足时留空。历史降水和流量统计是状态线索，统计关联本身不能证明产流机制或因果关系。'),
        ('5. 两种归一化结果',
         '最小最大归一化为 (x-x_min)/(x_max-x_min)，表示相对训练范围的位置。训练范围外的新值可能低于0或高于1，不截断。Z分数标准化为 (x-μ)/σ，表示距训练平均水平有多少个标准差；零附近表示接近均值，正负分别表示高于或低于均值。StandardScaler使用总体方差口径（ddof=0），不同于窗口样本标准差。标准化不保证数据服从正态分布，两种方法都对异常值敏感。常量列由sklearn处理，不进行手工除零。'),
        ('6. 拟合范围与特征选择边界',
         '两套Scaler复用有效训练样本（2000-01-08至2003-12-31）拟合的参数，应用到完整日期表，不利用测试期重新fit。窗口或滞后缺失仍传播为空值。归一化有助于对尺度敏感的模型训练；原始值、Min-Max值和Z分数值是同一信息的不同表示，通常按模型选择一种，而非同时全部输入。完整CSV保留当天径流及其变换作为观测结果；新版特征选择排除当天径流Discharge，保留历史径流滞后和窗口特征。新版排名为59个候选、57个保留特征；此前模型未重新训练，不能把已有模型指标当作该新子集的评估结果。'),
    ]
    for heading, text in sections:
        doc.add_heading(heading, level=3)
        doc.add_paragraph(text)
    examples = [
        ('编码数值示例', ['日期', '月份', '四季编码', '月份正弦编码', '月份余弦编码'], df.loc[df['日'] == 1].drop_duplicates('月份').head(12)),
        ('历史径流与窗口数值示例', ['日期', '当天径流量', '当天径流量_滞后1天', '当天径流量_滞后7天', '当天径流量_过去3天均值', '当天径流量_过去3天标准差'], df.iloc[7:12]),
        ('径流归一化数值示例', ['日期', '当天径流量', '当天径流量_最小最大归一化', '当天径流量_Z分数标准化'], df.iloc[7:12]),
    ]
    for title, cols, sample in examples:
        doc.add_heading(title, level=3)
        table = doc.add_table(rows=1, cols=len(cols))
        table.style = 'Table Grid'
        for cell, name in zip(table.rows[0].cells, cols):
            cell.text = name
        for _, row in sample.iterrows():
            for cell, col in zip(table.add_row().cells, cols):
                value = row[col]
                cell.text = '' if pd.isna(value) else (f'{value:.6f}' if isinstance(value, float) else str(value))
        for row in table.rows:
            for cell in row.cells:
                for p in cell.paragraphs:
                    for run in p.runs:
                        run.font.size = Pt(8)
        doc.add_paragraph('样例直接读取CSV；仅展示部分字段，浮点数显示6位小数，完整精度保存在CSV中。')
    # Move newly created content before the feature-selection section.
    body = doc.element.body
    added = [element for element in list(body)[start-1:] if element.tag.endswith('}sectPr') is False]
    anchor = next((p._p for p in doc.paragraphs if p.text.startswith(('2.3 子任务 8', '3.4 '))), None)
    if anchor is not None:
        for element in added:
            anchor.addprevious(element)
    out = path.with_stem(path.stem + '-特征数值说明补充版')
    doc.save(out)
    check = Document(out)
    text = '\n'.join(p.text for p in check.paragraphs)
    assert TITLE in text and all(title in text for title, _, _ in examples)
    assert len(check.tables) == len(Document(path).tables) + 3
    print(out, 'tables=', len(check.tables), 'images=', len(check.inline_shapes))

if __name__ == '__main__':
    supplement(ROOT / '《机器学习课程实践》第一次阶段验收报告（不含当天径流特征）.docx')
    supplement(ROOT / '《机器学习课程实践》实践报告-基于机器学习的径流预测.docx')
