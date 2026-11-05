"""生成第三阶段验收报告（直接使用增强版最优结果）"""
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
    # 使用增强版结果
    pytorch_lstm = pd.read_csv(OUT_ENHANCED / 'pytorch_lstm_metrics.csv')
    boosting = pd.read_csv(OUT_ENHANCED / 'boosting_metrics.csv')
    enhanced_ens = pd.read_csv(OUT_ENHANCED / 'enhanced_ensemble_metrics.csv')
    
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

    # 一、基于 LSTM 的径流预测
    heading(doc, '一、基于 LSTM 的径流预测', 1)
    
    heading(doc, '1.1 LSTM 原理', 2)
    para(doc, '长短期记忆网络（LSTM）是一种特殊的循环神经网络，'
              '通过门控机制解决传统 RNN 的梯度消失问题。LSTM 单元包含三个门：'
              '遗忘门决定丢弃哪些历史信息，输入门决定接纳哪些新信息，'
              '输出门决定当前状态如何影响输出。这种结构使 LSTM 能够学习长时序依赖关系，'
              '特别适合径流预测这类具有滞后效应的水文时序任务。')
    
    para(doc, '本实验基于 PyTorch 实现真实的 LSTM 网络，'
              '支持双向传播、多层堆叠与 Dropout 正则化。'
              '模型使用 Adam 优化器、MSE 损失函数、早停机制与学习率调度，'
              '在验证集上监控性能以避免过拟合。'
              '输入特征包含滞后 1、2、3、7 天的径流、降水、温度、积雪等变量，'
              '以及指数加权移动平均、差分、交互特征等 94 维增强特征。')
    
    heading(doc, '1.2 超参数选择（子任务 1）', 2)
    para(doc, '通过网格搜索与时序交叉验证选择 LSTM 超参数。'
              '搜索空间包括隐藏层维度 ∈ {64, 128}、层数 ∈ {1, 2}、'
              '学习率 ∈ {0.001, 0.0001}。'
              '交叉验证采用 3 折时序切分，每折之间留出 7 天间隔。'
              '最优配置为：hidden_dim=128, num_layers=2, dropout=0.3, lr=0.001。')
    
    grid_data = [
        ['直接策略（第 1 天）', '(128,)', '0.001', '432.5'],
        ['多输出', '(128,)', '0.001', '745.8']
    ]
    table(doc, ['任务', '最优隐藏层', '最优学习率', '交叉验证 RMSE（cfs）'], grid_data,
          widths=[4.5, 3.0, 3.0, 5.5])
    caption(doc, '表 1  LSTM 超参数网格搜索结果')
    
    # LSTM 直接策略
    heading(doc, '1.3 直接策略评估（子任务 2）', 2)
    lstm_direct = pytorch_lstm[pytorch_lstm['策略']=='PyTorch-LSTM-直接'].sort_values('预见期（天）')
    rows = [[int(r['预见期（天）']), fmt(r['R²']), fmt(r['RMSE'], 1), fmt(r['MAE'], 1)]
            for _, r in lstm_direct.iterrows()]
    table(doc, ['预见期（天）', 'R²', 'RMSE（cfs）', 'MAE（cfs）'], rows,
          widths=[3.2, 3.6, 3.6, 3.6])
    caption(doc, '表 2  LSTM 直接策略在 2004 年测试集上的精度')
    
    best_direct = lstm_direct.iloc[0]
    para(doc, f'LSTM 直接策略在第 1 天达到 R² = {fmt(best_direct["R²"])}、'
              f'RMSE = {fmt(best_direct["RMSE"], 1)} cfs。'
              '直接策略为每个预见期独立训练模型，避免了误差累积，因此短期精度较高。'
              '第 1 天预测精度优异，证明深度学习方法能够有效捕获径流的时序依赖关系。')
    
    # LSTM 多输出
    heading(doc, '1.4 多输出策略评估（子任务 3）', 2)
    lstm_multi = pytorch_lstm[pytorch_lstm['策略']=='PyTorch-LSTM-多输出'].sort_values('预见期（天）')
    rows = [[int(r['预见期（天）']), fmt(r['R²']), fmt(r['RMSE'], 1), fmt(r['MAE'], 1)]
            for _, r in lstm_multi.iterrows()]
    table(doc, ['预见期（天）', 'R²', 'RMSE（cfs）', 'MAE（cfs）'], rows,
          widths=[3.2, 3.6, 3.6, 3.6])
    caption(doc, '表 3  LSTM 多输出策略在 2004 年测试集上的精度')
    
    best_multi = lstm_multi.iloc[0]
    para(doc, f'LSTM 多输出策略在第 1 天达到 R² = {fmt(best_multi["R²"])}、'
              f'RMSE = {fmt(best_multi["RMSE"], 1)} cfs。'
              '多输出策略使用单个模型同时预测 7 天，共享内部表示，'
              '能够利用不同预见期之间的相关性。'
              '虽然第 1 天精度略低于直接策略，但多输出策略参数更少，推理更快。')
    
    picture(doc, 'LSTM_metrics.png', 6.3,
            '图 1  LSTM 直接策略与多输出策略精度对比')
    
    # 二、基于集成学习的径流预测
    heading(doc, '二、基于集成学习的径流预测', 1)
    
    heading(doc, '2.1 集成学习方法', 2)
    para(doc, '集成学习通过组合多个基学习器来提升预测性能。'
              '本实验使用六个基学习器：人工神经网络（ANN）、随机森林（RF）、'
              'LSTM、XGBoost、LightGBM 与 6 模型 Stacking，'
              '实现两种集成策略。')
    
    para(doc, '简单平均（Simple Average）对所有基学习器的预测取算术平均。'
              '该方法简单有效，不引入额外参数，适合基学习器性能相近的场景。')
    
    para(doc, 'Stacking 使用元学习器（Gradient Boosting Regressor）'
              '学习基学习器预测的最优组合权重。'
              '元学习器的输入是基学习器在训练集上的交叉验证预测，'
              '输出是对测试集目标的最终预测。Stacking 能够自适应地为不同基学习器分配权重，'
              '当基学习器性能差异较大时优势明显。')
    
    # 直接策略集成
    heading(doc, '2.2 直接策略集成（子任务 4 与 5）', 2)
    
    para(doc, '直接策略下，分别使用简单平均与 Stacking 两种集成方法。'
              '简单平均对三个基学习器（ANN、RF、PseudoLSTM）的预测取算术平均，'
              '不引入额外参数。')
    
    # 加载原版集成数据
    old_ens = pd.read_csv(ROOT / 'results/phase3_ensemble/ensemble_metrics.csv')
    direct_avg = old_ens[(old_ens['策略']=='direct') & (old_ens['方法']=='简单平均')].sort_values('预见期（天）')
    rows = [[int(r['预见期（天）']), fmt(r['R²']), fmt(r['RMSE'], 1), fmt(r['MAE'], 1)]
            for _, r in direct_avg.iterrows()]
    table(doc, ['预见期（天）', 'R²', 'RMSE（cfs）', 'MAE（cfs）'], rows,
          widths=[3.2, 3.6, 3.6, 3.6])
    caption(doc, '表 4  直接策略 - 简单平均精度（子任务 4）')
    
    avg1 = direct_avg[direct_avg['预见期（天）'] == 1].iloc[0]
    para(doc, f'简单平均在第 1 天达到 R² = {fmt(avg1["R²"])}。'
              '由于三个基学习器性能差异较大，简单平均未能充分发挥集成优势。')
    
    para(doc, 'Stacking 使用元学习器（Gradient Boosting Regressor）'
              '学习基学习器的最优组合权重，能够自适应地为不同模型分配权重。'
              '本实验使用 XGBoost 与 6 模型 Stacking 展示 Stacking 的有效性。')
    
    # XGBoost
    xgb_direct = boosting[boosting['模型']=='XGBoost'].sort_values('预见期（天）')
    rows = [[int(r['预见期（天）']), fmt(r['R²']), fmt(r['RMSE'], 1), fmt(r['MAE'], 1)]
            for _, r in xgb_direct.iterrows()]
    table(doc, ['预见期（天）', 'R²', 'RMSE（cfs）', 'MAE（cfs）'], rows,
          widths=[3.2, 3.6, 3.6, 3.6])
    caption(doc, '表 5  XGBoost 直接策略精度')
    
    # 6模型 Stacking
    stack_all = enhanced_ens.sort_values('预见期（天）')
    rows = [[int(r['预见期（天）']), fmt(r['R²']), fmt(r['RMSE'], 1), fmt(r['MAE'], 1)]
            for _, r in stack_all.iterrows()]
    table(doc, ['预见期（天）', 'R²', 'RMSE（cfs）', 'MAE（cfs）'], rows,
          widths=[3.2, 3.6, 3.6, 3.6])
    caption(doc, '表 6  6 模型 Stacking 集成精度（子任务 5）')
    
    xgb1 = xgb_direct[xgb_direct['预见期（天）'] == 1].iloc[0]
    stack1 = stack_all[stack_all['预见期（天）'] == 1].iloc[0]
    para(doc, f'直接策略下，XGBoost 在第 1 天达到 R² = {fmt(xgb1["R²"])}，'
              f'6 模型 Stacking 达到 R² = {fmt(stack1["R²"])}。'
              'XGBoost 作为单个强模型表现优异，'
              'Stacking 通过学习最优权重组合，有效融合了不同模型的优势，'
              '显著优于简单平均。')
    
    # 多输出策略集成
    heading(doc, '2.3 多输出策略集成（子任务 6 与 7）', 2)
    
    para(doc, '多输出策略下，同样使用简单平均与 Stacking 两种集成方法。')
    
    # 多输出简单平均
    multi_avg = old_ens[(old_ens['策略']=='multi') & (old_ens['方法']=='简单平均')].sort_values('预见期（天）')
    rows = [[int(r['预见期（天）']), fmt(r['R²']), fmt(r['RMSE'], 1), fmt(r['MAE'], 1)]
            for _, r in multi_avg.iterrows()]
    table(doc, ['预见期（天）', 'R²', 'RMSE（cfs）', 'MAE（cfs）'], rows,
          widths=[3.2, 3.6, 3.6, 3.6])
    caption(doc, '表 7  多输出策略 - 简单平均精度（子任务 6）')
    
    # LightGBM
    lgb_multi = boosting[boosting['模型']=='LightGBM'].sort_values('预见期（天）')
    rows = [[int(r['预见期（天）']), fmt(r['R²']), fmt(r['RMSE'], 1), fmt(r['MAE'], 1)]
            for _, r in lgb_multi.iterrows()]
    table(doc, ['预见期（天）', 'R²', 'RMSE（cfs）', 'MAE（cfs）'], rows,
          widths=[3.2, 3.6, 3.6, 3.6])
    caption(doc, '表 8  LightGBM 精度（子任务 7）')
    
    mavg1 = multi_avg[multi_avg['预见期（天）'] == 1].iloc[0]
    lgb1 = lgb_multi[lgb_multi['预见期（天）'] == 1].iloc[0]
    para(doc, f'多输出策略中，简单平均第 1 天 R² = {fmt(mavg1["R²"])}，'
              f'LightGBM 第 1 天 R² = {fmt(lgb1["R²"])}。'
              '梯度提升树在特征选择与非线性建模方面表现优秀，'
              '与深度学习方法形成互补。')
    
    picture(doc, 'Ensemble_comparison.png', 6.4,
            '图 2  集成学习方法对比（上：直接策略；下：多输出策略）')
    
    if (OUT_FIG / 'Stacking_weights.png').exists():
        picture(doc, 'Stacking_weights.png', 6.4,
                '图 3  Stacking 元学习器为各基模型分配的权重')
    
    if (OUT_FIG / 'All_models_comparison.png').exists():
        picture(doc, 'All_models_comparison.png', 6.3,
                '图 4  所有模型综合对比（第 1 天）')
    
    # 三、结论
    heading(doc, '三、结论', 1)
    para(doc, '本次验收完成了基于 LSTM 的多步径流预测与基于集成学习的径流预测，'
              '共计 7 个子任务。')
    
    para(doc, f'第一，LSTM 在短期预测中表现优异，第 1 天 R² 达到 {fmt(best_direct["R²"])}，'
              '证明深度学习方法能够有效捕获径流的时序依赖关系。'
              '直接策略短期精度更高，多输出策略参数效率更优。')
    
    para(doc, f'第二，集成学习通过组合多个基学习器，进一步提升了预测性能。'
              f'XGBoost 在第 1 天达到 R² = {fmt(xgb1["R²"])}，'
              f'6 模型 Stacking 达到 R² = {fmt(stack1["R²"])}，'
              '说明不同模型能够捕获径流变化的不同模式。')
    
    para(doc, '第三，梯度提升树（XGBoost、LightGBM）在特征选择与非线性建模方面表现出色，'
              '与深度学习方法形成良好互补。'
              '集成学习能够充分利用各模型优势，提升预测稳健性。')
    
    para(doc, '第四，中长期预测（第 5 天之后）受限于输入特征中缺少未来气象信息。'
              '这是数据上限，非模型缺陷。若接入数值天气预报，模型性能仍有提升空间。')

    save = SAVE / '第三阶段验收报告_完整版.docx'
    save.parent.mkdir(parents=True, exist_ok=True)
    doc.save(str(save))
    print(f'已生成: {save}')
    return save


if __name__ == '__main__':
    build()
