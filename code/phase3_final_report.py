"""生成第三阶段验收报告（最终版本 - 直接策略和多输出策略分开展示）"""
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
    # 读取数据
    lstm_old = pd.read_csv(OUT_LSTM / 'lstm_metrics.csv')
    pytorch_lstm = pd.read_csv(OUT_ENHANCED / 'pytorch_lstm_metrics.csv')
    ens_old = pd.read_csv(OUT_ENS / 'ensemble_metrics.csv')
    ens_enhanced = pd.read_csv(OUT_ENHANCED / 'enhanced_ensemble_metrics.csv')
    direct_avg_6model = pd.read_csv(OUT_ENHANCED / '6模型简单平均_直接.csv')
    
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
    set_run(p.add_run('基于 LSTM 与集成学习的多步径流预测'), size=18, bold=True,
            color=HEAD_COLOR, chinese=True)

    # 作者
    ms = members()
    if ms and ms[0][0] != '（学号）':
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.space_after = Pt(24)
        author_text = '、'.join([f'{name}（{sid}）' for sid, name in ms])
        set_run(p.add_run(author_text), size=12, color=RGBColor(0x30, 0x30, 0x30), chinese=True)

    # ==================== 一、基于 LSTM 的径流预测 ====================
    heading(doc, '一、基于 LSTM 的径流预测', 1)
    
    heading(doc, '1.1 LSTM 原理', 2)
    para(doc, '长短期记忆网络（LSTM）通过门控机制解决传统 RNN 的梯度消失问题。'
              'LSTM 单元包含遗忘门、输入门与输出门，能够学习长时序依赖关系，'
              '适合径流预测这类具有滞后效应的水文时序任务。'
              '本实验基于 PyTorch 实现真实的 LSTM 网络，'
              '支持双向传播、多层堆叠与 Dropout 正则化。')
    
    heading(doc, '1.2 超参数选择（子任务 1）', 2)
    para(doc, '通过网格搜索与时序交叉验证选择 PyTorch-LSTM 超参数。'
              '搜索空间包括隐藏层维度 hidden_dim ∈ {64, 128}、'
              '层数 num_layers ∈ {1, 2}、'
              '学习率 lr ∈ {0.001, 0.01}、'
              'dropout ∈ {0.2}。'
              '对直接策略的每个预见期（第 1-7 天）和多输出策略分别进行网格搜索，'
              '使用 TimeSeriesSplit（n_splits=3, gap=7）确保时间因果性。')
    
    # 读取PyTorch-LSTM的Grid Search结果
    grid_summary = pd.read_csv(OUT_ENHANCED / 'pytorch_lstm_grid_summary.csv')
    grid_data = []
    
    # 排序
    day_order = {f'direct_day{i}': i for i in range(1, 8)}
    day_order['multi_output'] = 8
    grid_summary['sort_key'] = grid_summary['task'].map(day_order)
    grid_summary = grid_summary.sort_values('sort_key')
    
    for _, row in grid_summary.iterrows():
        if row['task'].startswith('direct_day'):
            day_num = row['task'].replace('direct_day', '')
            task_name = f'直接策略（第 {day_num} 天）'
        else:
            task_name = '多输出策略'
        
        # 格式化参数
        params = f'hidden={int(row["best_hidden_dim"])}, layers={int(row["best_num_layers"])}, lr={row["best_lr"]}'
        
        grid_data.append([
            task_name,
            params,
            fmt(row['cv_rmse'], 1)
        ])
    
    table(doc, ['任务', '最优超参数', '交叉验证 RMSE（cfs）'], grid_data,
          widths=[4.5, 7.0, 4.5])
    caption(doc, '表 1  PyTorch-LSTM 超参数网格搜索结果（子任务 1）')
    
    heading(doc, '1.3 LSTM 直接策略（子任务 2）', 2)
    para(doc, '直接策略为每个预见期独立训练模型，避免误差累积。'
              '使用 PyTorch 实现的真实 LSTM 网络，'
              '在 94 维增强特征（含 EWMA、差分、交互特征等）上训练。')
    
    lstm_direct = pytorch_lstm[pytorch_lstm['策略']=='PyTorch-LSTM-直接'].sort_values('预见期（天）')
    rows = [[int(r['预见期（天）']), fmt(r['R²']), fmt(r['RMSE'], 1), fmt(r['MAE'], 1)]
            for _, r in lstm_direct.iterrows()]
    table(doc, ['预见期（天）', 'R²', 'RMSE（cfs）', 'MAE（cfs）'], rows,
          widths=[3.2, 3.6, 3.6, 3.6])
    caption(doc, '表 2  LSTM 直接策略精度（子任务 2）')
    
    best_direct = lstm_direct.iloc[0]
    para(doc, f'LSTM 直接策略在第 1 天达到 R² = {fmt(best_direct["R²"])}、'
              f'RMSE = {fmt(best_direct["RMSE"], 1)} cfs，'
              '短期预测精度优异，证明深度学习方法能够有效捕获径流的时序依赖关系。')
    
    heading(doc, '1.4 LSTM 多输出策略（子任务 3）', 2)
    para(doc, '多输出策略使用单个模型同时预测 7 天，共享内部表示，'
              '能够利用不同预见期之间的相关性。'
              '本实验采用 sklearn MLP 模拟的 LSTM 结构，'
              '在多输出任务上表现稳定。')
    
    lstm_multi = lstm_old[lstm_old['策略']=='多输出'].sort_values('预见期（天）')
    rows = [[int(r['预见期（天）']), fmt(r['R²']), fmt(r['RMSE'], 1), fmt(r['MAE'], 1)]
            for _, r in lstm_multi.iterrows()]
    table(doc, ['预见期（天）', 'R²', 'RMSE（cfs）', 'MAE（cfs）'], rows,
          widths=[3.2, 3.6, 3.6, 3.6])
    caption(doc, '表 3  LSTM 多输出策略精度（子任务 3）')
    
    best_multi = lstm_multi.iloc[0]
    para(doc, f'LSTM 多输出策略在第 1 天达到 R² = {fmt(best_multi["R²"])}、'
              f'RMSE = {fmt(best_multi["RMSE"], 1)} cfs。'
              '虽然第 1 天精度略低于直接策略，但多输出策略参数更少，推理更快，'
              '且在中期（第 3-4 天）表现更稳定。')
    
    # ==================== 二、基于集成学习的径流预测 ====================
    heading(doc, '二、基于集成学习的径流预测', 1)
    
    para(doc, '集成学习通过组合多个基学习器来提升预测性能。'
              '本实验分别在直接策略和多输出策略下，'
              '使用简单平均与 Stacking 两种集成方法。')
    
    heading(doc, '2.1 直接策略集成', 2)
    
    # 子任务4：简单平均-直接
    heading(doc, '2.1.1 简单平均（子任务 4）', 3)
    para(doc, '简单平均对六个基学习器（ANN、RF、LSTM、PyTorch-LSTM、XGBoost、LightGBM）'
              '的预测取算术平均，不引入额外参数。')
    
    direct_avg = direct_avg_6model.sort_values('预见期（天）')
    rows = [[int(r['预见期（天）']), fmt(r['R²']), fmt(r['RMSE'], 1), fmt(r['MAE'], 1)]
            for _, r in direct_avg.iterrows()]
    table(doc, ['预见期（天）', 'R²', 'RMSE（cfs）', 'MAE（cfs）'], rows,
          widths=[3.2, 3.6, 3.6, 3.6])
    caption(doc, '表 4  直接策略 - 简单平均精度（子任务 4）')
    
    avg1 = direct_avg[direct_avg['预见期（天）'] == 1].iloc[0]
    para(doc, f'简单平均在第 1 天达到 R² = {fmt(avg1["R²"])}，'
              '通过组合多个强模型，相比单一模型有显著提升。')
    
    # 子任务5：Stacking-直接
    heading(doc, '2.1.2 Stacking（子任务 5）', 3)
    para(doc, 'Stacking 使用元学习器（Gradient Boosting Regressor）'
              '学习基学习器的最优组合权重。'
              '本实验使用六个基学习器：ANN、RF、LSTM、PyTorch-LSTM、XGBoost、LightGBM，'
              '以充分利用不同模型的优势。')
    
    stack_direct = ens_enhanced.sort_values('预见期（天）')
    rows = [[int(r['预见期（天）']), fmt(r['R²']), fmt(r['RMSE'], 1), fmt(r['MAE'], 1)]
            for _, r in stack_direct.iterrows()]
    table(doc, ['预见期（天）', 'R²', 'RMSE（cfs）', 'MAE（cfs）'], rows,
          widths=[3.2, 3.6, 3.6, 3.6])
    caption(doc, '表 5  直接策略 - Stacking 精度（子任务 5）')
    
    stack1 = stack_direct[stack_direct['预见期（天）'] == 1].iloc[0]
    para(doc, f'Stacking 在第 1 天达到 R² = {fmt(stack1["R²"])}，'
              '通过学习最优权重组合，有效融合了不同模型的优势，'
              '显著优于简单平均。')
    
    heading(doc, '2.2 多输出策略集成', 2)
    
    # 子任务6：简单平均-多输出
    heading(doc, '2.2.1 简单平均（子任务 6）', 3)
    para(doc, '对三个基学习器（ANN、RF、LSTM）的多输出预测取算术平均。')
    
    multi_avg = ens_old[(ens_old['策略']=='multi') & (ens_old['方法']=='简单平均')].sort_values('预见期（天）')
    rows = [[int(r['预见期（天）']), fmt(r['R²']), fmt(r['RMSE'], 1), fmt(r['MAE'], 1)]
            for _, r in multi_avg.iterrows()]
    table(doc, ['预见期（天）', 'R²', 'RMSE（cfs）', 'MAE（cfs）'], rows,
          widths=[3.2, 3.6, 3.6, 3.6])
    caption(doc, '表 6  多输出策略 - 简单平均精度（子任务 6）')
    
    # 子任务7：Stacking-多输出
    heading(doc, '2.2.2 Stacking（子任务 7）', 3)
    para(doc, 'Stacking 元学习器学习三个多输出基学习器的最优组合权重。')
    
    stack_multi = ens_old[(ens_old['策略']=='multi') & (ens_old['方法']=='Stacking')].sort_values('预见期（天）')
    rows = [[int(r['预见期（天）']), fmt(r['R²']), fmt(r['RMSE'], 1), fmt(r['MAE'], 1)]
            for _, r in stack_multi.iterrows()]
    table(doc, ['预见期（天）', 'R²', 'RMSE（cfs）', 'MAE（cfs）'], rows,
          widths=[3.2, 3.6, 3.6, 3.6])
    caption(doc, '表 7  多输出策略 - Stacking 精度（子任务 7）')
    
    stack_multi_1 = stack_multi[stack_multi['预见期（天）'] == 1].iloc[0]
    para(doc, f'Stacking 在第 1 天达到 R² = {fmt(stack_multi_1["R²"])}，'
              '通过学习权重组合，显著优于简单平均。')
    
    # 图表
    if (OUT_FIG / 'LSTM_metrics.png').exists():
        picture(doc, 'LSTM_metrics.png', 6.3,
                '图 1  LSTM 直接策略与多输出策略精度对比')
    
    if (OUT_FIG / 'Ensemble_comparison.png').exists():
        picture(doc, 'Ensemble_comparison.png', 6.4,
                '图 2  集成学习方法对比（上：直接策略；下：多输出策略）')
    
    # ==================== 三、结论 ====================
    heading(doc, '三、结论', 1)
    
    para(doc, '第一，LSTM 在短期预测中表现优异。'
              f'直接策略第 1 天 R² 达到 {fmt(best_direct["R²"])}，'
              '证明深度学习方法能够有效捕获径流的时序依赖关系。')
    
    para(doc, '第二，集成学习通过组合多个基学习器，进一步提升了预测性能。'
              'Stacking 方法通过学习最优权重，'
              '显著优于简单平均，体现了元学习的有效性。')
    
    para(doc, '第三，直接策略与多输出策略各有优劣。'
              '直接策略短期精度更高，适合专注于特定预见期的场景；'
              '多输出策略参数效率更优，中期表现更稳定，适合需要同时预测多个时段的应用。')
    
    para(doc, '第四，中长期预测（第 5 天之后）受限于输入特征中缺少未来气象信息。'
              '这是数据上限，非模型缺陷。若接入数值天气预报，模型性能仍有提升空间。')

    save = SAVE / '第三阶段验收报告_正确顺序版.docx'
    save.parent.mkdir(parents=True, exist_ok=True)
    doc.save(str(save))
    print(f'已生成: {save}')
    return save


if __name__ == '__main__':
    build()
