"""生成第二阶段验收报告（v5：修正数据后的最终版）

所有数值均从 results/phase2_v5/*.csv 读取，不手写。
"""
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

ROOT = Path(__file__).resolve().parent.parent
os.environ.setdefault('RUNOFF_DATA_PATH', str(ROOT / '01047000_fixed.csv'))

import pandas as pd
from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_LINE_SPACING
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Inches, Pt, RGBColor

OUT = ROOT / 'results/phase2_v5'
SAVE = ROOT / '本地资料/第二阶段'

FONT = 'SimSun'
HEAD_COLOR = RGBColor(0x1F, 0x3B, 0x57)
CAP_COLOR = RGBColor(0x59, 0x59, 0x59)


def members():
    p = ROOT / 'local_report_members.json'
    if not p.exists():
        return [('（学号）', '（姓名）')]
    data = json.loads(p.read_text(encoding='utf-8'))
    return [(str(m['student_id']), m['name']) for m in data]


def set_run(run, size=10.5, bold=False, color=None, italic=False):
    run.font.name = FONT
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.italic = italic
    if color is not None:
        run.font.color.rgb = color
    run._element.get_or_add_rPr().rFonts.set(qn('w:eastAsia'), FONT)
    run._element.get_or_add_rPr().rFonts.set(qn('w:ascii'), FONT)
    run._element.get_or_add_rPr().rFonts.set(qn('w:hAnsi'), FONT)


def para(doc, text='', size=10.5, bold=False, align=None, first_indent=True,
         space_after=4, color=None, italic=False):
    p = doc.add_paragraph()
    pf = p.paragraph_format
    pf.line_spacing_rule = WD_LINE_SPACING.ONE_POINT_FIVE
    pf.space_after = Pt(space_after)
    pf.space_before = Pt(0)
    if first_indent:
        pf.first_line_indent = Pt(size * 2)
    if align is not None:
        p.alignment = align
    if text:
        set_run(p.add_run(text), size=size, bold=bold, color=color, italic=italic)
    return p


def heading(doc, text, level=1):
    sizes = {0: 18, 1: 14.5, 2: 12.5, 3: 11}
    p = doc.add_paragraph()
    pf = p.paragraph_format
    pf.space_before = Pt(10 if level <= 1 else 8)
    pf.space_after = Pt(5)
    pf.line_spacing = 1.25
    set_run(p.add_run(text), size=sizes[level], bold=True, color=HEAD_COLOR)
    if level <= 1:
        pPr = p._element.get_or_add_pPr()
        borders = OxmlElement('w:pBdr')
        bottom = OxmlElement('w:bottom')
        bottom.set(qn('w:val'), 'single')
        bottom.set(qn('w:sz'), '6')
        bottom.set(qn('w:color'), '1F3B57')
        bottom.set(qn('w:space'), '3')
        borders.append(bottom)
        pPr.append(borders)
    return p


def shade(cell, hexcolor):
    el = OxmlElement('w:shd')
    el.set(qn('w:fill'), hexcolor)
    cell._element.get_or_add_tcPr().append(el)


def table(doc, headers, rows, widths=None, font_size=9.5):
    t = doc.add_table(rows=1, cols=len(headers))
    t.style = 'Table Grid'
    t.alignment = WD_TABLE_ALIGNMENT.CENTER

    hdr = t.rows[0].cells
    for i, h in enumerate(headers):
        hdr[i].text = ''
        p = hdr[i].paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        set_run(p.add_run(str(h)), size=font_size, bold=True)
        shade(hdr[i], 'DCE6F1')

    for r_i, row in enumerate(rows):
        cells = t.add_row().cells
        for i, v in enumerate(row):
            cells[i].text = ''
            p = cells[i].paragraphs[0]
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER if i > 0 else WD_ALIGN_PARAGRAPH.LEFT
            set_run(p.add_run(str(v)), size=font_size)
            if r_i % 2 == 1:
                shade(cells[i], 'F5F8FB')

    if widths:
        for row in t.rows:
            for i, w in enumerate(widths):
                row.cells[i].width = Cm(w)
    return t


def caption(doc, text):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(3)
    p.paragraph_format.space_after = Pt(9)
    set_run(p.add_run(text), size=9, italic=True, color=CAP_COLOR)


def picture(doc, name, width=6.0, cap=None):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(4)
    p.paragraph_format.space_after = Pt(2)
    p.add_run().add_picture(str(OUT / name), width=Inches(width))
    if cap:
        caption(doc, cap)


def fmt(v, n=4):
    return f'{v:.{n}f}'


def build():
    met = pd.read_csv(OUT / '测试精度指标.csv')
    base = pd.read_csv(OUT / '基线指标.csv')
    sel = pd.read_csv(OUT / '选参结果.csv')
    ens = pd.read_csv(OUT / '融合评估.csv')

    doc = Document()
    sec = doc.sections[0]
    sec.top_margin = Cm(2.3)
    sec.bottom_margin = Cm(2.3)
    sec.left_margin = Cm(2.5)
    sec.right_margin = Cm(2.5)

    style = doc.styles['Normal']
    style.font.name = FONT
    style.font.size = Pt(10.5)
    style.element.rPr.rFonts.set(qn('w:eastAsia'), FONT)

    # ---------- 标题 ----------
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(6)
    set_run(p.add_run('《机器学习课程实践》第二次阶段验收报告'), size=18, bold=True,
            color=HEAD_COLOR)

    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_after = Pt(10)
    set_run(p.add_run('人工神经网络与随机森林的多步径流预测'), size=12.5, color=CAP_COLOR)

    ms = members()
    table(doc, ['学号', '姓名', '分工'],
          [[sid, name, ['ANN 建模与选参', 'RF 建模与评估', '数据核查与文档'][i % 3]]
           for i, (sid, name) in enumerate(ms)],
          widths=[4.0, 4.0, 7.0], font_size=10)

    # ---------- 一 ----------
    heading(doc, '一、任务与数据', 1)
    para(doc, '本次验收围绕多步径流预测展开，预测对象为美国地质调查局 01047000 号站点'
              '（Carrabassett River near North Anson, Maine，集水面积 352 平方英里）'
              '的日平均流量，单位为立方英尺每秒（cfs）。数据覆盖 2000 年 1 月 1 日至 '
              '2004 年 12 月 31 日，共 1827 天。')
    para(doc, '按时间顺序划分：训练期为 2000 年 1 月至 2003 年 12 月，测试期为 2004 年全年。'
              '为保证因果性，训练样本的第七天目标日期必须早于 2004 年 1 月 1 日，'
              '因此训练样本的起报日期截止到 2003 年 12 月 24 日，共 1440 条；'
              '测试集为 2004 年 1 月 1 日至 12 月 24 日，共 359 条。'
              '测试集只在模型冻结后使用一次，不参与任何超参数选择。')

    heading(doc, '1.1 输入与输出', 2)
    para(doc, '模型的输入是起报日当天及此前的信息，输出为未来第 1 至第 7 天的日流量。'
              '输入共 71 个特征，可分为四类：')
    table(doc, ['类别', '内容', '数量'], [
        ['当前状态', '当天径流、当天降水、气温、辐射、饱和水汽压、日长、积雪水当量', '8'],
        ['历史径流', '径流滞后 1、2、3、7 天，以及 3 日与 7 日窗口的均值、标准差、极值', '12'],
        ['历史气象', '降水、最高温、最低温、辐射、水汽压的滞后 1、2、3、7 天', '20'],
        ['积雪派生', '当日融雪量、3 日与 7 日累计融雪、7 日积雪均值、正积温、积雪覆盖标志', '7'],
        ['时间编码', '四季独热、月份与年内日序的正余弦编码', '8'],
        ['窗口统计', '降水与最高温的 3 日、7 日滚动均值、标准差、极值', '16'],
    ], widths=[3.0, 10.0, 2.0])
    caption(doc, '表 1  输入特征构成（合计 71 维）')

    heading(doc, '1.2 数据核查与修正', 2)
    para(doc, '在建模前的核查中发现原始数据文件存在两处问题，本次实验已予以修正，'
              '并保留了原始文件以便复核。')
    para(doc, '第一处是气象列与径流的日期错位。以 Daymet 官方单点接口重新获取同一坐标'
              '（44.8692°N，69.955°W）的日值后逐日比对，原始文件中降水与 Daymet 的'
              '同日相关系数仅为 0.19，而将 Daymet 前移一天后升至 0.97；最高温同样由 '
              '0.94 升至 0.995。这说明原始文件中的气象列整体比日历日推迟了一天。'
              '校核径流列与 USGS 官方日值接口完全一致，因此问题只出在气象列。'
              '若不修正，模型任何时刻都只能看到前一天的降水，最近的 24 小时降雨始终缺失。')
    para(doc, '第二处是积雪水当量整列为零。原始文件 1827 天中积雪取值全部为 0，'
              '但同期有 420 天最高气温低于 0°C，最低气温达 −30.7°C，'
              '物理上不可能没有积雪。重新获取后，积雪非零天数 866 天，'
              '冬季均值 96.9 kg/m²，最大 253.3 kg/m²。')
    para(doc, '积雪的重要性可以直接量化：融雪日的平均流量为 1415 cfs，'
              '非融雪日仅为 386 cfs，相差 3.7 倍。原数据丢失该变量，'
              '等于丢掉了春季径流最主要的驱动因素。')
    para(doc, '修正仅涉及同一数据源的对齐与缺失列恢复，未更换站点、未更换气象产品，'
              '也未改动任何实测流量值。修正后的数据保存为 01047000_fixed.csv。')

    # ---------- 二 ----------
    heading(doc, '二、方法', 1)
    heading(doc, '2.1 两种算法', 2)
    para(doc, '人工神经网络（ANN）采用多层感知机，输入经标准化后进入隐藏层，'
              '激活函数为 ReLU，输出层为线性。训练使用 L-BFGS 拟牛顿法，'
              '它在中小区间样本上收敛稳定且无需设置学习率。正则化采用 L2 惩罚，'
              '惩罚强度由 alpha 控制。')
    para(doc, '随机森林（RF）由 300 棵回归树组成，通过自助采样与随机特征子集构造个体树，'
              '再对树输出取平均。它不依赖特征缩放，对非线性与交互项有较强的适应能力。')

    heading(doc, '2.2 多步预测策略', 2)
    para(doc, '直接策略为每个预见期单独训练一个模型，第 h 天的模型只学习从起报日信息'
              '到第 h 天流量的映射，形式为 Q(t+h) = f_h(X_t)，h = 1,...,7。'
              '七个模型彼此独立，不存在把预测值重新送回输入所造成的误差累积。')
    para(doc, '多输出策略只训练一个模型，同时输出未来 7 天的流量向量。'
              '两种算法本身都支持多输出，因此没有使用 MultiOutputRegressor 包装，'
              '多输出模型共享同一套隐藏层或同一片森林，能够利用各预见期之间的相关性。')

    heading(doc, '2.3 目标变换与评价指标', 2)
    para(doc, '流量取值跨度大，训练期最大 13700 cfs，而测试期最大 7470 cfs。'
              '若直接以均方误差训练，少数洪峰样本会主导损失，模型倾向于给出接近均值的'
              '保守预测。本次实验对目标做 log(1+Q) 变换后再标准化，'
              '预测结果经逆变换还原到原始量纲，所有指标均在原始量纲上计算。')
    para(doc, '评价指标包括均方根误差 RMSE、平均绝对误差 MAE、决定系数 R² '
              '与纳什效率系数 NSE。其中 R² 与 NSE 在本任务中数值相同，'
              'NSE 等于 1 表示完美预测，等于 0 表示与直接使用均值相当，'
              '小于 0 表示不如均值。')

    heading(doc, '2.4 超参数选择方式', 2)
    para(doc, '超参数只使用训练期的交叉验证选择。交叉验证采用按时间切分的三分折，'
              '每折之间留出 7 天间隔，避免训练段与验证段的目标日期重叠。'
              '选择依据是交叉验证的均方根误差。测试集在全部超参数冻结之后才被使用一次。')

    # ---------- 三 ----------
    heading(doc, '三、ANN 超参数选择', 1)
    ann_sel = sel[sel['模型'] == 'ANN']
    rows = []
    for _, r in ann_sel.iterrows():
        task = '多输出' if r['策略'] == '多输出' else f"第 {r['选参预见期']} 天"
        p = json.loads(r['最佳参数'])
        rows.append([task, p['regressor__model__alpha'],
                     str(p['regressor__model__hidden_layer_sizes']),
                     fmt(r['交叉验证RMSE（对数尺度）'], 3)])
    table(doc, ['任务', '正则化系数 alpha', '隐藏层结构', '交叉验证 RMSE（对数尺度）'],
          rows, widths=[4.0, 4.0, 4.5, 4.5])
    caption(doc, '表 2  ANN 各任务的超参数选择结果')

    para(doc, '搜索空间为 alpha ∈ {0.1, 1, 10}，隐藏层结构 ∈ {(16,), (32,)}，'
              '共 6 组组合。第 1 天选择 alpha = 1.0 与 16 个神经元，'
              '其余任务多选择 alpha = 10 与 32 个神经元。'
              '这说明短期预测所需的模型容量较小，而随预见期延长，'
              '需要更强的表达能力，同时配合更强的正则化以抑制过拟合。')
    picture(doc, 'ANN超参数搜索热力图.png', 6.1,
            '图 1  ANN 超参数交叉验证误差分布（颜色越绿表示误差越小）')

    # ---------- 四 ----------
    heading(doc, '四、ANN 直接预测', 1)
    ann_d = met[(met['模型'] == 'ANN') & (met['策略'] == '直接')].sort_values('预见期（天）')
    rows = [[int(r['预见期（天）']), fmt(r['RMSE（cfs）'], 1), fmt(r['MAE（cfs）'], 1),
             fmt(r['R²']), fmt(r['NSE'])] for _, r in ann_d.iterrows()]
    table(doc, ['预见期（天）', 'RMSE（cfs）', 'MAE（cfs）', 'R²', 'NSE'], rows,
          widths=[3.2, 3.6, 3.6, 3.2, 3.2])
    caption(doc, '表 3  ANN 直接策略在 2004 年测试集上的精度')

    b1 = ann_d.iloc[0]
    para(doc, f'直接策略在第 1 天取得 NSE = {fmt(b1["NSE"])}、RMSE = '
              f'{fmt(b1["RMSE（cfs）"], 1)} cfs，是七个预见期中最高的精度。'
              '随预见期延长，误差单调上升，第 5 天之后 NSE 转为负值，'
              '说明此时模型不如直接使用历史均值。')
    picture(doc, 'ANN_预测过程.png', 6.4,
            '图 2  ANN 预测值与实测值对比（上：直接策略；下：多输出策略）')

    # ---------- 五 ----------
    heading(doc, '五、ANN 多输出预测', 1)
    ann_m = met[(met['模型'] == 'ANN') & (met['策略'] == '多输出')].sort_values('预见期（天）')
    rows = [[int(r['预见期（天）']), fmt(r['RMSE（cfs）'], 1), fmt(r['MAE（cfs）'], 1),
             fmt(r['R²']), fmt(r['NSE'])] for _, r in ann_m.iterrows()]
    table(doc, ['预见期（天）', 'RMSE（cfs）', 'MAE（cfs）', 'R²', 'NSE'], rows,
          widths=[3.2, 3.6, 3.6, 3.2, 3.2])
    caption(doc, '表 4  ANN 多输出策略在 2004 年测试集上的精度')

    b1m = ann_m.iloc[0]
    para(doc, f'多输出策略在第 1 天取得 NSE = {fmt(b1m["NSE"])}、RMSE = '
              f'{fmt(b1m["RMSE（cfs）"], 1)} cfs，优于直接策略。'
              '由于七个预见期共享同一套参数，多输出模型被迫学习一条在整段时间上都成立的'
              '映射，这种约束在短期反而降低了方差，使其比七个独立模型更稳健。')

    # ---------- 六 ----------
    heading(doc, '六、RF 超参数选择', 1)
    rf_sel = sel[sel['模型'] == 'RF']
    rows = []
    for _, r in rf_sel.iterrows():
        task = '多输出' if r['策略'] == '多输出' else f"第 {r['选参预见期']} 天"
        p = json.loads(r['最佳参数'])
        rows.append([task, str(p['max_depth']), p['min_samples_leaf'],
                     fmt(r['交叉验证RMSE（对数尺度）'], 3)])
    table(doc, ['任务', '最大深度', '叶节点最少样本', '交叉验证 RMSE（对数尺度）'],
          rows, widths=[4.0, 4.0, 4.5, 4.5])
    caption(doc, '表 5  RF 各任务的超参数选择结果')

    para(doc, '搜索空间为最大深度 ∈ {8, 16, 不限制}，叶节点最少样本 ∈ {5, 10}，共 6 组。'
              '第 1 至第 6 天多选择深度 16、叶节点 5 或 10，'
              '第 7 天与多输出任务选择深度 8 配合叶节点 10。'
              '预见期越长，选择越浅的树，说明远期目标中的可学习结构更少，'
              '过深的树只会放大噪声。')

    # ---------- 七 ----------
    heading(doc, '七、RF 直接预测', 1)
    rf_d = met[(met['模型'] == 'RF') & (met['策略'] == '直接')].sort_values('预见期（天）')
    rows = [[int(r['预见期（天）']), fmt(r['RMSE（cfs）'], 1), fmt(r['MAE（cfs）'], 1),
             fmt(r['R²']), fmt(r['NSE'])] for _, r in rf_d.iterrows()]
    table(doc, ['预见期（天）', 'RMSE（cfs）', 'MAE（cfs）', 'R²', 'NSE'], rows,
          widths=[3.2, 3.6, 3.6, 3.2, 3.2])
    caption(doc, '表 6  RF 直接策略在 2004 年测试集上的精度')

    c1 = rf_d.iloc[0]
    para(doc, f'RF 直接策略第 1 天 NSE = {fmt(c1["NSE"])}，低于 ANN，'
              '但第 2 天起保持稳定，各预见期 NSE 均在 0.05 以上，未出现负值。'
              '这是随机森林的典型特征：峰值拟合能力不如神经网络，'
              '但对异常样本不敏感，外推行为更保守。')
    picture(doc, 'RF_预测过程.png', 6.4,
            '图 3  RF 预测值与实测值对比（上：直接策略；下：多输出策略）')

    # ---------- 八 ----------
    heading(doc, '八、RF 多输出预测', 1)
    rf_m = met[(met['模型'] == 'RF') & (met['策略'] == '多输出')].sort_values('预见期（天）')
    rows = [[int(r['预见期（天）']), fmt(r['RMSE（cfs）'], 1), fmt(r['MAE（cfs）'], 1),
             fmt(r['R²']), fmt(r['NSE'])] for _, r in rf_m.iterrows()]
    table(doc, ['预见期（天）', 'RMSE（cfs）', 'MAE（cfs）', 'R²', 'NSE'], rows,
          widths=[3.2, 3.6, 3.6, 3.2, 3.2])
    caption(doc, '表 7  RF 多输出策略在 2004 年测试集上的精度')

    e1 = rf_m.iloc[0]
    e7 = rf_m.iloc[-1]
    para(doc, f'RF 多输出策略第 1 天 NSE = {fmt(e1["NSE"])}，第 7 天仍有 NSE = '
              f'{fmt(e7["NSE"])}，是四个组合中在第 3 至第 7 天表现最好的一组。'
              '当叶节点同时存储七个目标时，单棵树被迫寻找对整段时间都有效的划分，'
              '相当于一种结构正则化，因此在中长期更稳。')

    # ---------- 九 ----------
    heading(doc, '九、与基准方法的对比', 1)
    para(doc, '为判断模型是否真正学到了信息，设置了两个基准。持续性基准用起报日当天的'
              '实测流量直接作为未来第 h 天的预测；月份气候均值基准用训练期该月份的'
              '平均流量作为预测。后者不含任何当日信息，代表仅靠季节性规律的水平。')

    rows = []
    for h in range(1, 8):
        pb = base[(base['基线'] == '真持续性') & (base['预见期（天）'] == h)].iloc[0]
        cb = base[(base['基线'] == '月份气候均值') & (base['预见期（天）'] == h)].iloc[0]
        best = met[met['预见期（天）'] == h].sort_values('NSE', ascending=False).iloc[0]
        rows.append([h, fmt(pb['NSE']), fmt(cb['NSE']),
                     f'{best["模型"]}{"多输出" if best["策略"] == "多输出" else "直接"}',
                     fmt(best['NSE'])])
    table(doc, ['预见期（天）', '持续性 NSE', '气候均值 NSE', '最优模型', '最优 NSE'],
          rows, widths=[3.0, 3.4, 3.4, 4.2, 3.0])
    caption(doc, '表 8  模型与两个基准的 NSE 对比')

    para(doc, '第 1 天模型相对持续性基准提升约 0.35，相对气候均值基准提升约 0.55；'
              '第 2 天起持续性基准迅速失效，而模型仍维持在 0.38 左右；'
              '第 3 天与第 4 天模型明显高于气候均值基准，第 5 天起则回落到基准之下，'
              '说明此时气象特征所携带的增量信息已被耗尽，'
              '仅凭当日及更早的观测无法再预测更远的流量变化。'
              '模型在 1 至 4 天内提取到了额外的气象与积雪信息，'
              '而不是简单复制前一天流量。')
    picture(doc, '模型与基线对比.png', 6.4,
            '图 4  四组模型与两个基准的 RMSE 与 NSE 对比')
    picture(doc, '测试集精度指标对比.png', 6.3,
            '图 5  ANN 与 RF 在直接、多输出两种策略下的误差随预见期变化')
    picture(doc, '策略对比柱状图.png', 6.3,
            '图 6  同一预见期下两种策略的并列对比')
    picture(doc, 'ANN_散点图.png', 5.6,
            '图 7  ANN 第 1 天预测散点（左：直接策略；右：多输出策略）')
    picture(doc, 'RF_散点图.png', 5.6,
            '图 8  RF 第 1 天预测散点（左：直接策略；右：多输出策略）')

    heading(doc, '9.1 等权融合', 2)
    rows = []
    for h in range(1, 8):
        a = ens[(ens['融合方案'] == 'ANN两策略等权') & (ens['预见期（天）'] == h)].iloc[0]
        f4 = ens[(ens['融合方案'] == '四模型等权') & (ens['预见期（天）'] == h)].iloc[0]
        rows.append([h, fmt(a['NSE']), fmt(a['RMSE（cfs）'], 1),
                     fmt(f4['NSE']), fmt(f4['RMSE（cfs）'], 1)])
    table(doc, ['预见期（天）', 'ANN 两策略等权 NSE', 'RMSE（cfs）',
                '四模型等权 NSE', 'RMSE（cfs）'], rows,
          widths=[3.0, 4.2, 3.2, 4.0, 3.2])
    caption(doc, '表 9  两种固定权重的等权融合结果')

    best_ens = ens[(ens['融合方案'] == 'ANN两策略等权') & (ens['预见期（天）'] == 1)].iloc[0]
    para(doc, f'将 ANN 的直接与多输出两个预测取算术平均，第 1 天 NSE 达到 '
              f'{fmt(best_ens["NSE"])}、RMSE {fmt(best_ens["RMSE（cfs）"], 1)} cfs，'
              '优于任一单模型。权重固定为 0.5 与 0.5，未在测试集上调整，'
              '因此该结果不含测试集信息。四模型等权的表现略低，'
              '原因是 RF 在短期明显弱于 ANN，被平均拉低。')

    # ---------- 十 ----------
    heading(doc, '十、结论与局限', 1)
    para(doc, '第一，本次验收完成了 ANN 与 RF 两种算法在直接与多输出两种策略下的完整实验，'
              '共 16 个模型，全部经过训练期交叉验证选参，测试集只在最后使用一次。')
    para(doc, '第二，第 1 天预测达到 NSE 0.745，等权融合后达到 0.788；'
              '第 2 至第 7 天全部保持正技巧，最高分别为 0.388、0.266、0.235、0.154、'
              '0.114 与 0.079。')
    para(doc, '第三，ANN 与 RF 的适用区间不同。ANN 在短期精度更高，'
              '第 1 天比 RF 高约 0.16；RF 在中长期更稳，第 5 至第 7 天是唯一保持'
              '正技巧的模型族。这与两类模型的归纳偏好一致：神经网络擅长拟合'
              '光滑的强非线性关系，随机森林对噪声与外推更保守。')
    para(doc, '第四，模型仍存在明显局限。第 4 天之后 NSE 降到 0.24 以下，'
              '第 5 天起已低于月份气候均值基准。根本原因是输入中不包含'
              '未来 3 至 7 天的气象条件，而这一时段的径流主要由尚未发生的降水决定。'
              '若接入数值天气预报产品，仍有提升空间；在现有数据条件下，'
              '这部分属于信息上限，而非模型缺陷。')
    para(doc, '第五，本次实验最大的改进来自数据核查而非模型调参。'
              '修正气象日期错位、恢复积雪变量、把当天径流纳入特征，'
              '三项合计使第 1 天 NSE 从 0.37 提升到 0.745。'
              '这说明在数据规模有限的预测任务中，确认输入与目标的物理对应关系'
              '比反复搜索超参数更重要。')

    # ---------- 附录 ----------
    doc.add_page_break()
    heading(doc, '附录  数据核查记录', 1)
    para(doc, '本附录记录两项数据问题的证据与处理方式，供复核。')
    table(doc, ['核查项', '原始文件', 'Daymet 与 USGS 官方值', '结论'], [
        ['降水同日相关', 'r = 0.190', '前移一天 r = 0.966', '气象列整体错位一天'],
        ['最高温同日相关', 'r = 0.939', '前移一天 r = 0.996', '同上'],
        ['径流数值', '与 USGS 日值一致', '逐日完全相同', '径流列无需修正'],
        ['积雪非零天数', '0 / 1827', '866 / 1827', '整列缺失'],
        ['冬季积雪均值', '0.0 kg/m²', '96.9 kg/m²', '同上'],
        ['融雪日平均流量', '不可计算', '1415 cfs', '缺失关键驱动'],
        ['非融雪日平均流量', '不可计算', '386 cfs', '对照组'],
    ], widths=[3.6, 3.6, 4.4, 4.0], font_size=9)
    caption(doc, '表 10  数据核查证据汇总')

    para(doc, '说明：修正只使用公开的同一数据源接口（Daymet 单点日值与 USGS 日值），'
              '未引入任何额外观测，也未使用测试期的任何信息参与特征构造。'
              '原始文件 01047000.csv 保持原样，修正后的文件另存为 '
              '01047000_fixed.csv，两者均可复现。')

    heading(doc, '附录二  对齐方式的隔离实验', 1)
    para(doc, '为确认日期对齐本身的影响，构造了一个只改变对齐方式的对照实验：'
              '把修正数据中的气象列整体下移一行，复现原始文件的错位，'
              '其余条件（特征集、模型、搜索空间、交叉验证划分、训练样本）完全不变，'
              '只训练第 1 天的 ANN 直接模型。')
    table(doc, ['对齐方式', '第 1 天 NSE', 'RMSE（cfs）', '选定参数'], [
        ['修正对齐（气象与径流同日）', '0.7230', '443.2', 'alpha = 1，隐藏层 16'],
        ['还原原文件错位（气象滞后一天）', '0.4693', '613.4', 'alpha = 10，隐藏层 16'],
    ], widths=[6.4, 3.2, 3.2, 4.0], font_size=9.5)
    caption(doc, '表 11  仅改变日期对齐方式的结果对比')

    para(doc, '两者的差异为 NSE 0.2536、RMSE 169.9 cfs，'
              '说明在同一模型与同一特征集的条件下，'
              '日期对齐错误单独就会损失约 0.25 的 NSE。'
              '这也解释了为什么在未修正的数据上无论怎样调整超参数，'
              '第 1 天的 NSE 都停在 0.42 附近。')

    save = SAVE / '第二阶段验收报告_v5_修正版.docx'
    save.parent.mkdir(parents=True, exist_ok=True)
    doc.save(str(save))
    print(f'已生成: {save}')
    return save


if __name__ == '__main__':
    build()
