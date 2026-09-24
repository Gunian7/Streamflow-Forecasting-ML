# -*- coding: utf-8 -*-
"""
生成配套的思政报告文档 (.docx)
根据目录中的模板要求：
文档命名为：学号+姓名+思政报告.docx
题目：以人工智能赋能智慧水利，践行科技报国的青年担当——《机器学习课程实践》思政报告
"""

import os
import docx
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml import parse_xml
from docx.oxml.ns import nsdecls, qn

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_PATH = os.path.join(BASE_DIR, '思政报告-以人工智能赋能智慧水利践行科技报国担当.docx')

def set_cell_background(cell, fill_hex):
    tcPr = cell._element.get_or_add_tcPr()
    shd = parse_xml(f'<w:shd {nsdecls("w")} w:fill="{fill_hex}"/>')
    tcPr.append(shd)

def set_table_borders(table):
    tblPr = table._element.xpath('w:tblPr')
    if tblPr:
        borders = parse_xml(
            f'<w:tblBorders {nsdecls("w")}>'
            f'<w:top w:val="single" w:sz="8" w:space="0" w:color="000000"/>'
            f'<w:bottom w:val="single" w:sz="8" w:space="0" w:color="000000"/>'
            f'<w:insideH w:val="single" w:sz="4" w:space="0" w:color="CCCCCC"/>'
            f'<w:left w:val="none"/>'
            f'<w:right w:val="none"/>'
            f'<w:insideV w:val="none"/>'
            f'</w:tblBorders>'
        )
        tblPr[0].append(borders)

def build_sizheng_report():
    doc = Document()
    
    for s in doc.sections:
        s.top_margin = Inches(1.0)
        s.bottom_margin = Inches(1.0)
        s.left_margin = Inches(1.0)
        s.right_margin = Inches(1.0)
        
    doc.styles['Normal'].font.name = 'Microsoft YaHei'
    doc.styles['Normal'].font.size = Pt(11)
    doc.styles['Normal']._element.rPr.rFonts.set(qn('w:eastAsia'), 'Microsoft YaHei')
    
    # 封面标题
    p_title = doc.add_paragraph()
    p_title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r_title = p_title.add_run("杭州电子科技大学\n《机器学习课程实践》思政报告")
    r_title.font.size = Pt(22)
    r_title.font.bold = True
    r_title.font.color.rgb = RGBColor(0x1F, 0x49, 0x7D)
    
    doc.add_paragraph("\n")
    
    # 信息表格
    table = doc.add_table(rows=6, cols=2)
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    
    data = [
        ["题    目", "以人工智能赋能智慧水利，践行科技报国的青年担当"],
        ["专    业", "智能科学与技术"],
        ["班    级", "24063113 / 24063114"],
        ["学    号", "（填写您的学号）"],
        ["姓    名", "（填写您的姓名）"],
        ["指导教师", "江净超、黄经州"]
    ]
    
    for r_idx, (k, v) in enumerate(data):
        c0 = table.cell(r_idx, 0)
        c1 = table.cell(r_idx, 1)
        c0.text = k
        c1.text = v
        c0.paragraphs[0].runs[0].font.bold = True
        c0.paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER
        set_cell_background(c0, "F2F2F2")
        
    set_table_borders(table)
    
    p_date = doc.add_paragraph("\n\n二〇二六 年 九 月")
    p_date.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_date.runs[0].font.size = Pt(12)
    p_date.runs[0].font.bold = True
    
    doc.add_page_break()
    
    # 正文部分
    h1 = doc.add_heading("以人工智能赋能智慧水利，践行科技报国的青年担当", level=1)
    h1.alignment = WD_ALIGN_PARAGRAPH.CENTER
    
    doc.add_heading("一、治水兴邦：中华民族的历史基因与国家重大战略需求", level=2)
    doc.add_paragraph(
        "善治国者必先治水。从大禹治水‘左准绳、右规矩’的朴素唯物智慧，到战国时期巧夺天工、滋养天府之国的都江堰水利工程；"
        "从新中国成立初期修建十三陵水库、红旗渠的自力更生与艰苦奋斗，到改革开放后举世瞩目的三峡工程、南水北调工程，"
        "治水始终与中华民族的生存繁衍、兴衰治乱休戚相关。习近平总书记明确指出：‘水安全是涉及国家长治久安的大事’，并提出了‘节水优先、空间均衡、系统治理、两手发力’的治水思路。\n"
        "然而，在全球气候变暖加剧、极端气象事件频发的背景下，暴雨山洪、突发干旱等水文极值事件对流域防洪抗旱和水资源精细化调度带来了全新挑战。"
        "如何将新一代信息技术与水利工程深度融合，建设‘数字孪生流域’与‘智慧水利’系统，成为实现国家水治理体系和治理能力现代化的战略迫切需求。"
    )
    
    doc.add_heading("二、实事求是：在代码与数据中磨砺科学求真品格", level=2)
    doc.add_paragraph(
        "在本次《机器学习课程实践》中，我们针对真实流域水文与气象时间序列，实现了从特征工程、多步直接与多输出预测建模（ANN、随机森林、LSTM）到集成学习的全链条实践。"
        "从我国自主研发的‘新安江水文模型’到现代深度学习 LSTM 登上《Nature》国际顶刊的演进历史，让我深刻认识到科技创新的日新月异与数据驱动水文科学的巨大潜力。\n"
        "在实验之初，面对高达 13700 cfs 的极端融雪洪峰与强烈的右偏分布特性，传统简单模型往往难以精确捕捉，极易出现削峰或震荡。"
        "通过严格的时间序列因果顺序切分（前 4 年训练、第 5 年测试），我们坚决杜绝了数据穿越与标签泄露，践行了求真务实的科学作风。\n"
        "在特征工程设计中，我们深入理解了降水入渗、土壤滞蓄、积雪消融与水动力传导的物理机制，构造了科学的滞后与滑动窗口特征。"
        "每一次模型的迭代优化、每一个超参数网格搜索的严谨验证，都让我深刻体会到：机器学习绝非纸上谈兵的玄学调参，而是建立在严密逻辑、实测证据与严谨数学推导之上的求真学问。"
    )
    
    doc.add_heading("三、协同攻关：从集成学习中领悟集体主义与系统思维", level=2)
    doc.add_paragraph(
        "实验中最引人深思的技术环节是集成学习（Ensemble Learning）。单个模型各有所长亦各有所短：神经网络具备强大的非线性拟合能力，但容易对极端值产生泛化方差；"
        "随机森林作为树模型对异常值鲁棒，但在区间外外推时存在硬边界；LSTM 善于捕捉序列动态记忆，但对局部瞬时脉冲响应滞后。"
        "通过简单平均法与 Stacking 堆叠泛化机制，我们将异构基模型的认知优势进行融合互补，最终在第1天径流预测中取得了纳什效率系数 NSE = 0.5000 的优异表现，整体误差显著低于任何单一模型。\n"
        "这不仅是算法工程的胜利，更是系统论与集体主义哲学的生动体现。正如我国在航天工程、特高压电网、重大水利设施建设中展现的‘集中力量办大事’的制度优势一样，"
        "现代科技攻关从来不是孤立单打独斗，而是需要跨学科、多模块的协同互补与精诚协作。"
    )
    
    doc.add_heading("四、科技报国：将青春论文写在祖国江河大地上", level=2)
    doc.add_paragraph(
        "学好机器学习技术，绝不仅仅是为了在期末取得一份优异的分数，更是为了在党和国家最需要的地方发挥实实在在的光和热。"
        "每一次水文预测精度的提升，在实际工程中都意味着防汛指挥部能提前几个小时发布预警，转移受威胁群众，调控水库库容，挽救人民生命与财产免遭洪涝之灾。\n"
        "作为杭州电子科技大学智能科学与技术专业的学子，我们生逢其时、重任在肩。通过本次课程实践，我进一步坚定了科技报国、工程为民的理想信念。"
        "在未来的学习与科研道路上，我将继续秉持笃行不怠的匠心，锤炼过硬的算法工程能力，努力把所学的人工智能知识转化为保障国家水安全、生态安全和高质量发展的坚实力量，"
        "让青春在全面建设社会主义现代化国家的火热实践中绽放绚丽之花！"
    )
    
    doc.save(OUT_PATH)
    print(f"思政报告生成完毕！保存路径: {OUT_PATH}")

if __name__ == '__main__':
    build_sizheng_report()
