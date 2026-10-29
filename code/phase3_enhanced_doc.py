"""生成第三阶段增强版验收报告（包含精度提升结果）"""
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd
from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_LINE_SPACING
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Inches, Pt, RGBColor

ROOT = Path(__file__).resolve().parent.parent
OUT_LSTM = ROOT / 'results/phase3_lstm'
OUT_ENS = ROOT / 'results/phase3_ensemble'
OUT_ENHANCED = ROOT / 'results/phase3_enhanced'
OUT_FIG = ROOT / 'results/phase3_figures'
SAVE = ROOT / '本地资料/第三阶段'

FONT = 'Times New Roman'
FONT_CN = 'SimSun'
HEAD_COLOR = RGBColor(0x00, 0x00, 0x00)
CAP_COLOR = RGBColor(0x40, 0x40, 0x40)


def members():
    p = ROOT / 'local_report_members.json'
    if not p.exists():
        return [('（学号）', '（姓名）')]
    data = json.loads(p.read_text(encoding='utf-8'))
    return [(str(m['student_id']), m['name']) for m in data]


def set_run(run, size=10.5, bold=False, color=None, italic=False, chinese=False):
    font_name = FONT_CN if chinese else FONT
    run.font.name = font_name
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.italic = italic
    if color is not None:
        run.font.color.rgb = color
    run._element.get_or_add_rPr().rFonts.set(qn('w:eastAsia'), FONT_CN)
    run._element.get_or_add_rPr().rFonts.set(qn('w:ascii'), FONT)
    run._element.get_or_add_rPr().rFonts.set(qn('w:hAnsi'), FONT)


def para(doc, text='', size=10.5, bold=False, align=None, first_indent=True,
         space_after=4, color=None, chinese=True):
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
        set_run(p.add_run(text), size=size, bold=bold, color=color, chinese=chinese)
    return p


def heading(doc, text, level=1):
    sizes = {0: 16, 1: 14, 2: 12, 3: 11}
    p = doc.add_paragraph()
    pf = p.paragraph_format
    pf.space_before = Pt(12 if level <= 1 else 8)
    pf.space_after = Pt(6)
    pf.line_spacing = 1.25
    if level <= 1:
        p.alignment = WD_ALIGN_PARAGRAPH.LEFT
    set_run(p.add_run(text), size=sizes[level], bold=True, color=HEAD_COLOR, chinese=True)
    return p


def shade(cell, hexcolor):
    el = OxmlElement('w:shd')
    el.set(qn('w:fill'), hexcolor)
    cell._element.get_or_add_tcPr().append(el)


def table(doc, headers, rows, widths=None, font_size=10.5):
    t = doc.add_table(rows=1, cols=len(headers))
    t.style = 'Table Grid'
    t.alignment = WD_TABLE_ALIGNMENT.CENTER

    hdr = t.rows[0].cells
    for i, h in enumerate(headers):
        hdr[i].text = ''
        p = hdr[i].paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        set_run(p.add_run(str(h)), size=font_size, bold=True, chinese=True)

    for r_i, row in enumerate(rows):
        cells = t.add_row().cells
        for i, v in enumerate(row):
            cells[i].text = ''
            p = cells[i].paragraphs[0]
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            set_run(p.add_run(str(v)), size=font_size, chinese=True)

    if widths:
        for row in t.rows:
            for i, w in enumerate(widths):
                row.cells[i].width = Cm(w)
    
    # 三线表
    for row_idx, row in enumerate(t.rows):
        for cell in row.cells:
            tcPr = cell._element.get_or_add_tcPr()
            tcBorders = OxmlElement('w:tcBorders')
            
            if row_idx == 0:
                top = OxmlElement('w:top')
                top.set(qn('w:val'), 'single')
                top.set(qn('w:sz'), '12')
                top.set(qn('w:color'), '000000')
                tcBorders.append(top)
            
            if row_idx == 0:
                bottom = OxmlElement('w:bottom')
                bottom.set(qn('w:val'), 'single')
                bottom.set(qn('w:sz'), '6')
                bottom.set(qn('w:color'), '000000')
                tcBorders.append(bottom)
            
            if row_idx == len(t.rows) - 1:
                bottom = OxmlElement('w:bottom')
                bottom.set(qn('w:val'), 'single')
                bottom.set(qn('w:sz'), '12')
                bottom.set(qn('w:color'), '000000')
                tcBorders.append(bottom)
            
            for side in ['left', 'right']:
                border = OxmlElement(f'w:{side}')
                border.set(qn('w:val'), 'none')
                tcBorders.append(border)
            
            tcPr.append(tcBorders)
    
    return t


def caption(doc, text):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(3)
    p.paragraph_format.space_after = Pt(9)
    set_run(p.add_run(text), size=10, italic=False, color=CAP_COLOR, chinese=True)


def picture(doc, name, width=6.0, cap=None):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(4)
    p.paragraph_format.space_after = Pt(2)
    p.add_run().add_picture(str(OUT_FIG / name), width=Inches(width))
    if cap:
        caption(doc, cap)


def fmt(v, n=4):
    return f'{v:.{n}f}'


def build():
    # 读取所有结果
    lstm_metrics = pd.read_csv(OUT_LSTM / 'lstm_metrics.csv')
    ens_metrics = pd.read_csv(OUT_ENS / 'ensemble_metrics.csv')
    
    # 增强版结果
    pytorch_lstm = pd.read_csv(OUT_ENHANCED / 'pytorch_lstm_metrics.csv')
    boosting = pd.read_csv(OUT_ENHANCED / 'boosting_metrics.csv')
    enhanced_ens = pd.read_csv(OUT_ENHANCED / 'enhanced_ensemble_metrics.csv')
    ranking = pd.read_csv(OUT_ENHANCED / '精度排行榜_第1天.csv')
    
    doc = Document()
    sec = doc.sections[0]
    sec.top_margin = Cm(2.3)
    sec.bottom_margin = Cm(2.3)
    sec.left_margin = Cm(2.5)
    sec.right_margin = Cm(2.5)

    style = doc.styles['Normal']
    style.font.name = FONT
    style.font.size = Pt(10.5)
    style.element.rPr.rFonts.set(qn('w:eastAsia'), FONT_CN)

    # 标题
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(12)
    p.paragraph_format.space_after = Pt(18)
    set_run(p.add_run('基于深度学习与集成学习的多步径流预测'), size=18, bold=True,
            color=HEAD_COLOR, chinese=True)

    # 作者
    ms = members()
    if ms and ms[0][0] != '（学号）':
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.space_after = Pt(24)
        author_text = '、'.join([f'{name}（{sid}）' for sid, name in ms])
        set_run(p.add_run(author_text), size=12, color=RGBColor(0x30, 0x30, 0x30), chinese=True)

    # 摘要
    heading(doc, '摘要', 0)
    para(doc, '本报告基于 LSTM 与集成学习实现多步径流预测。第三阶段验收实现了 '
              'PseudoLSTM（sklearn MLP 模拟）、简单平均与 Stacking 集成，'
              '第1天预测达到 R²=0.778。后续优化采用真实 PyTorch LSTM、'
              'XGBoost、LightGBM 与增强特征工程（94维），'
              '并实现 6 模型非线性 Stacking 集成。最终 PyTorch-LSTM 直接策略第1天达到 '
              'R²=0.8328，超额完成目标（≥0.80），提升幅度 ΔR²=+0.0550。', 
         first_indent=False, space_after=12)

    # 一、验收任务完成情况
    heading(doc, '一、验收任务完成情况（7个子任务）', 1)
    
    para(doc, '按照验收要求完成 LSTM 建模与集成学习的全部 7 个子任务。')
    
    # 子任务汇总表
    tasks_data = [
        ['1', 'LSTM 超参数网格搜索', '✓', '直接: (64,) α=0.01 | 多输出: (64,) α=0.01'],
        ['2', 'LSTM 直接策略 (1-7天)', '✓', '第1天 R²=0.7778, RMSE=396.9'],
        ['3', 'LSTM 多输出策略 (1-7天)', '✓', '第1天 R²=0.7503, RMSE=420.7'],
        ['4', '直接策略-简单平均 (1-7天)', '✓', '第1天 R²=0.1873, RMSE=759.1'],
        ['5', '直接策略-Stacking (1-7天)', '✓', '第1天 R²=0.7712, RMSE=402.8'],
        ['6', '多输出-简单平均 (1-7天)', '✓', '第1天 R²=0.2242, RMSE=741.6'],
        ['7', '多输出-Stacking (1-7天)', '✓', '第1天 R²=0.7489, RMSE=421.9'],
    ]
    table(doc, ['子任务', '内容', '状态', '关键结果（第1天）'], tasks_data,
          widths=[1.8, 4.5, 1.5, 7.5])
    caption(doc, '表 1  验收任务完成情况')

    # 二、精度提升优化
    heading(doc, '二、精度提升优化', 1)
    
    heading(doc, '2.1 优化动机与目标', 2)
    para(doc, '验收完成后，为进一步提升预测精度，在原基础上实施四项优化：'
              '（1）使用真实 PyTorch LSTM 替代 MLP 模拟；'
              '（2）引入梯度提升树（XGBoost、LightGBM）；'
              '（3）增强特征工程（EWMA、差分、交互特征）；'
              '（4）6 模型非线性 Stacking 集成。'
              '优化目标为第1天 R² 从 0.778 提升至 ≥0.80。')
    
    heading(doc, '2.2 增强特征工程', 2)
    para(doc, '在原 71 维特征基础上新增 23 个特征，总计 94 维。'
              '新增特征包括：指数加权移动平均（EWMA，捕获短期趋势）、'
              '滞后差分（捕获变化率）、交互特征（降水×温度、积雪×正积温，体现物理意义）、'
              '滚动标准差（捕获波动性）、累积统计与比率特征。'
              '增强特征使训练集从 1440 条减少到 1434 条（因 rolling 和 diff 产生 NaN），'
              '测试集从 359 条减少到 353 条。')
    
    heading(doc, '2.3 真实 PyTorch LSTM', 2)
    para(doc, 'PseudoLSTM 本质是多层感知机，无真正的时序记忆机制。'
              '本次实现基于 PyTorch 的真实 LSTM，支持双向、多层、Dropout、'
              '早停与学习率调度。模型使用 Adam 优化器、MSE 损失、'
              '梯度裁剪，并在 20% 训练集尾段作验证集，早停耐心 15 轮。')
    
    pytorch_d1 = pytorch_lstm[(pytorch_lstm['策略']=='PyTorch-LSTM-直接') & 
                               (pytorch_lstm['预见期（天）']==1)].iloc[0]
    para(doc, f'PyTorch-LSTM 直接策略在第1天达到 R²={fmt(pytorch_d1["R²"])}、'
              f'RMSE={fmt(pytorch_d1["RMSE"], 1)} cfs，'
              '较 PseudoLSTM 提升 ΔR²=+0.0550，训练仅耗时 16 秒。')
    
    heading(doc, '2.4 梯度提升树（XGBoost 与 LightGBM）', 2)
    para(doc, '引入 XGBoost 3.2.0 和 LightGBM 4.7.0 作为补充基学习器。'
              '两者均使用 500 棵树、最大深度 6、学习率 0.05、L1 与 L2 正则化，'
              '并在验证集上早停。')
    
    xgb_d1 = boosting[(boosting['模型']=='XGBoost') & (boosting['预见期（天）']==1)].iloc[0]
    lgb_d1 = boosting[(boosting['模型']=='LightGBM') & (boosting['预见期（天）']==1)].iloc[0]
    para(doc, f'XGBoost 第1天 R²={fmt(xgb_d1["R²"])}，'
              f'LightGBM 第1天 R²={fmt(lgb_d1["R²"])}。'
              'XGBoost 单模型即超过 0.80 目标。')
    
    heading(doc, '2.5 6 模型增强集成', 2)
    para(doc, '集成 ANN、RF、PseudoLSTM、PyTorch-LSTM、XGBoost、LightGBM 六个基学习器。'
              '使用 GradientBoostingRegressor 作为非线性元学习器（100棵树、深度3、学习率0.05），'
              '替代原 Ridge 线性元学习器。'
              '元特征通过时间序列 3 折交叉验证生成，避免训练集泄漏。'
              '由于旧模型（ANN/RF/PseudoLSTM）用 71 维特征、新模型用 94 维，'
              '集成时按日期对齐并映射到各自特征空间。')
    
    enh_d1 = enhanced_ens[enhanced_ens['预见期（天）']==1].iloc[0]
    old_stack = ens_metrics[(ens_metrics['策略']=='direct') & (ens_metrics['方法']=='Stacking') & 
                            (ens_metrics['预见期（天）']==1)].iloc[0]
    para(doc, f'6 模型 Stacking 第1天 R²={fmt(enh_d1["R²"])}、'
              f'RMSE={fmt(enh_d1["RMSE"], 1)} cfs，'
              f'较原 3 模型 Stacking（R²={fmt(old_stack["R²"])}）提升 '
              f'ΔR²=+{fmt(enh_d1["R²"]-old_stack["R²"])}。')

    # 三、第1天精度排行榜
    heading(doc, '三、第1天精度排行榜', 1)
    
    para(doc, '汇总所有模型在 2004 年测试集第1天的预测精度，按 R² 降序排列。')
    
    rows = []
    for _, r in ranking.iterrows():
        rows.append([r['模型'], fmt(r['R²']), fmt(r['RMSE'], 1), 
                     fmt(r['MAE'], 1), r['类型']])
    table(doc, ['模型', 'R²', 'RMSE（cfs）', 'MAE（cfs）', '类型'], rows,
          widths=[4.5, 2.5, 2.8, 2.8, 3.5])
    caption(doc, '表 2  第1天精度排行榜（按 R² 降序）')
    
    best = ranking.iloc[0]
    baseline = ranking[ranking['模型']=='PseudoLSTM'].iloc[0]
    para(doc, f'最优模型为 {best["模型"]}，R²={fmt(best["R²"])}，'
              f'较基线（PseudoLSTM，R²={fmt(baseline["R²"])}）'
              f'提升 ΔR²=+{fmt(best["R²"]-baseline["R²"])}，'
              f'相对提升 {(best["R²"]-baseline["R²"])/baseline["R²"]*100:.1f}%。'
              f'已超额完成目标（R² ≥ 0.80），超出 {(best["R²"]-0.80)*100:.1f}%。')

    # 四、1-7天整体表现
    heading(doc, '四、1-7天整体表现', 1)
    
    heading(doc, '4.1 PyTorch-LSTM 直接策略', 2)
    pytorch_direct = pytorch_lstm[pytorch_lstm['策略']=='PyTorch-LSTM-直接'].sort_values('预见期（天）')
    rows = [[int(r['预见期（天）']), fmt(r['R²']), fmt(r['RMSE'], 1), fmt(r['MAE'], 1)]
            for _, r in pytorch_direct.iterrows()]
    table(doc, ['预见期（天）', 'R²', 'RMSE（cfs）', 'MAE（cfs）'], rows,
          widths=[3.2, 3.6, 3.6, 3.6])
    caption(doc, '表 3  PyTorch-LSTM 直接策略精度（1-7天）')
    
    heading(doc, '4.2 XGBoost', 2)
    xgb_all = boosting[boosting['模型']=='XGBoost'].sort_values('预见期（天）')
    rows = [[int(r['预见期（天）']), fmt(r['R²']), fmt(r['RMSE'], 1), fmt(r['MAE'], 1)]
            for _, r in xgb_all.iterrows()]
    table(doc, ['预见期（天）', 'R²', 'RMSE（cfs）', 'MAE（cfs）'], rows,
          widths=[3.2, 3.6, 3.6, 3.6])
    caption(doc, '表 4  XGBoost 精度（1-7天）')
    
    heading(doc, '4.3 6 模型增强集成', 2)
    enh_all = enhanced_ens.sort_values('预见期（天）')
    rows = [[int(r['预见期（天）']), fmt(r['R²']), fmt(r['RMSE'], 1), fmt(r['MAE'], 1)]
            for _, r in enh_all.iterrows()]
    table(doc, ['预见期（天）', 'R²', 'RMSE（cfs）', 'MAE（cfs）'], rows,
          widths=[3.2, 3.6, 3.6, 3.6])
    caption(doc, '表 5  6 模型增强集成精度（1-7天）')

    # 五、结论
    heading(doc, '五、结论', 1)
    
    para(doc, '第一，验收任务全部完成。实现了 LSTM 超参数搜索、'
              '直接与多输出策略、简单平均与 Stacking 集成，共 7 个子任务。')
    
    para(doc, '第二，精度提升目标超额完成。通过真实 PyTorch LSTM、'
              '梯度提升树、增强特征工程与 6 模型集成，'
              '第1天 R² 从 0.778 提升至 0.833，超出目标 0.80 约 4.1%。')
    
    para(doc, '第三，短期预测精度显著提升，中长期受限于数据。'
              '第1天提升 0.055，第2天提升约 0.04，'
              '但第5天之后因输入特征中缺少未来气象信息，精度提升受限。'
              '这是信息上限，非模型缺陷，若接入数值天气预报可进一步提升。')
    
    para(doc, '第四，真实深度学习框架效率显著。PyTorch-LSTM 训练仅耗时 16 秒，'
              '达到 0.833 精度，证明在特征工程充分的情况下，'
              '真实 LSTM 比 MLP 模拟更有效。')
    
    para(doc, '第五，集成学习具有稳健性。6 模型 Stacking 虽未达到单模型峰值，'
              '但在中长期（第2-4天）表现更平稳，适合需要同时预报多个时段的业务场景。')

    # 附录
    doc.add_page_break()
    heading(doc, '附录  技术实现细节', 1)
    
    heading(doc, 'A.1 增强特征列表', 2)
    para(doc, '在原 71 维基础上新增 23 个特征：', first_indent=False)
    features_list = [
        'Discharge_ewm3, Prcp_ewm3, Tmax_ewm3 (指数加权移动平均 span=3)',
        'Discharge_ewm7, Prcp_ewm7, Tmax_ewm7 (指数加权移动平均 span=7)',
        'Discharge_diff1, Prcp_diff1, Swe_diff1 (1 阶差分)',
        'Discharge_diff3, Prcp_diff3, Swe_diff3 (3 阶差分)',
        'Prcp_x_Tmax (降水与温度交互)',
        'Swe_x_PDD (积雪与正积温交互)',
        'Prcp_x_Swe (降水与积雪交互)',
        'Discharge_std_3, Prcp_std_3 (3 日滚动标准差)',
        'Discharge_std_7, Prcp_std_7 (7 日滚动标准差)',
        'Prcp_cum3, Prcp_cum7, PDD_cum3 (累积统计)',
        'Q_to_P_ratio (径流降水比)'
    ]
    for f in features_list:
        p = doc.add_paragraph(style='List Bullet')
        p.paragraph_format.left_indent = Pt(24)
        p.paragraph_format.first_line_indent = Pt(-12)
        set_run(p.add_run(f), size=10, chinese=True)
    
    heading(doc, 'A.2 模型超参数', 2)
    hyperparams = [
        ['PyTorch-LSTM', 'hidden_dim=128, num_layers=2, dropout=0.3, lr=0.001'],
        ['XGBoost', 'n_estimators=500, max_depth=6, lr=0.05, subsample=0.8'],
        ['LightGBM', 'n_estimators=500, max_depth=6, lr=0.05, subsample=0.8'],
        ['6模型 Stacking 元学习器', 'GBR(n_estimators=100, max_depth=3, lr=0.05)']
    ]
    table(doc, ['模型', '超参数'], hyperparams, widths=[4.0, 11.0], font_size=9.5)
    caption(doc, '表 6  增强版模型超参数')

    save = SAVE / '第三阶段验收报告_含精度提升.docx'
    save.parent.mkdir(parents=True, exist_ok=True)
    doc.save(str(save))
    print(f'已生成: {save}')
    return save


if __name__ == '__main__':
    build()
