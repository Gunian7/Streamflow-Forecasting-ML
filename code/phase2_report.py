"""Generate a compact phase-2 report from real experiment artifacts."""
from pathlib import Path
import json
import pandas as pd
from docx import Document
from docx.shared import Cm, Pt, RGBColor
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from report_members import member_fields
from report_typography import formal_typography
from phase2_presentation import add_method_explanation, add_strategy_analysis, add_search_figure

ROOT=Path(__file__).resolve().parent.parent
OUT=ROOT/'results/phase2_v2'

def main():
    m=pd.read_csv(OUT/'预测精度指标.csv')
    params=pd.read_csv(OUT/'最佳超参数.csv')
    meta=json.loads((OUT/'实验配置.json').read_text(encoding='utf-8'))
    assert len(m)==28 and len(params)==16
    doc=Document()
    section=doc.sections[0]
    section.top_margin=section.bottom_margin=Cm(1.6)
    section.left_margin=section.right_margin=Cm(1.7)
    normal=doc.styles['Normal']
    normal.font.name='Calibri'; normal.font.size=Pt(9)
    normal._element.get_or_add_rPr().rFonts.set(qn('w:eastAsia'),'Microsoft YaHei')
    normal.paragraph_format.space_after=Pt(5)
    normal.paragraph_format.line_spacing=1.15
    for name,size in [('Heading 1',15),('Heading 2',11)]:
        s=doc.styles[name]; s.font.size=Pt(size); s.font.color.rgb=RGBColor.from_string('234C59')
        s.paragraph_format.space_before=Pt(10); s.paragraph_format.space_after=Pt(5)
    doc.add_heading('杭州电子科技大学\n《机器学习课程实践》阶段验收报告',0)
    doc.add_paragraph('第二次阶段验收：人工神经网络与随机森林径流预测')
    ids,names=member_fields()
    doc.add_paragraph('成员：'+'；'.join(n+'（'+i+'）' for n,i in zip(names.splitlines(),ids.splitlines())))
    doc.add_paragraph('时间：10月22日 13:30—16:50　地点：第12教研楼202')
    def table(frame):
        tb=doc.add_table(rows=1,cols=len(frame.columns)); tb.style='Table Grid'
        for cell,c in zip(tb.rows[0].cells,frame.columns): cell.text=str(c)
        repeat=OxmlElement('w:tblHeader'); tb.rows[0]._tr.get_or_add_trPr().append(repeat)
        for row in frame.itertuples(index=False,name=None):
            for cell,v in zip(tb.add_row().cells,row): cell.text=f'{v:.4f}' if isinstance(v,float) else str(v)
        for i,row in enumerate(tb.rows):
            row._tr.get_or_add_trPr().append(OxmlElement('w:cantSplit'))
            for cell in row.cells:
                if i==0:
                    sh=OxmlElement('w:shd'); sh.set(qn('w:fill'),'DFEBEE'); cell._tc.get_or_add_tcPr().append(sh)
                for p in cell.paragraphs:
                    p.paragraph_format.space_after=Pt(2)
                    for r in p.runs: r.font.size=Pt(8)
    doc.add_heading('内容概览',1)
    table(pd.DataFrame([['1','适用方法','模型特点与适用条件'],['2','ANN超参数优化','结构、搜索空间、最优参数与收敛记录'],['3','ANN直接预测','七个预见期指标与预测过程'],['4','ANN多输出预测','七个预见期指标与预测过程'],['5','随机森林超参数优化','结构、搜索空间与最优参数'],['6','随机森林直接预测','七个预见期指标与预测过程'],['7','随机森林多输出预测','七个预见期指标与预测过程']],columns=['编号','研究内容','展示结果']))
    doc.add_heading('实验数据与预测设置',1)
    doc.add_paragraph(f"训练起报日期：{meta['训练起']}至{meta['训练止']}（{meta['训练样本']}条）；测试起报日期：{meta['测试起']}至{meta['测试止']}（{meta['测试样本']}条）。共使用{meta['特征数量']}个候选输入，包括季节四列独热编码、月份/日序周期编码、气象观测、历史滞后及窗口统计，不含当天径流和Swe。为避免监督筛选引入交叉验证信息，本实验使用全部候选输入，不复用第一阶段的监督筛选子集。")
    doc.add_paragraph('预测目标为起报日期后第1至第7天径流。直接策略分别训练7个模型，多输出策略使用一个模型联合输出7天。训练样本的第7天目标日期也必须早于2004年；交叉验证采用3折扩展窗口，训练/验证之间留7行间隔。ANN的输入与输出标准化均在每个训练折内部拟合；随机森林使用原始尺度。')
    table(pd.read_csv(OUT/'交叉验证划分.csv'))
    doc.add_heading('1｜适用于径流预测的方法',1)
    table(pd.DataFrame([['ANN','学习输入到径流的非线性关系；对尺度、初始化敏感'],['随机森林','利用多棵回归树集成；不依赖变量缩放'],['支持向量回归','可用核函数描述非线性关系；需调整核与正则化'],['梯度提升树','逐步拟合残差；需控制树复杂度与学习率'],['LSTM','处理历史序列动态；需更多训练和结构选择']],columns=['方法','特点']))
    for family,base in [('ANN',2),('RF',5)]:
        doc.add_heading(f'{base}｜{family} 网格搜索与超参数',1)
        doc.add_paragraph('每个直接模型和多输出模型分别搜索，以3折验证RMSE均值最小为选择依据。最优参数指候选空间内交叉验证误差最低的配置。')
        add_method_explanation(doc,family)
        if family=='ANN':
            doc.add_paragraph('候选：隐藏层(16,)或(32,)，L2系数0.1、1或10，优化器L-BFGS，随机种子42。每个模型6组候选、3折时序验证。迭代与函数调用保护上限分别为5000和100000，停止容差为1e-5。L-BFGS不使用固定学习率。')
        else:
            doc.add_paragraph('候选：最大深度8、16或不限制，叶节点最少样本1、3或5；固定300棵树、随机种子42。每个模型9组候选、3折验证，再在全部训练样本重拟合优选配置。')
        subset=params[params['模型']==family][['选参预见期','最佳参数','交叉验证RMSE']].copy()
        subset['选参预见期']=subset['选参预见期'].replace({'multi':'多输出1—7天'})
        table(subset)
        add_search_figure(doc,OUT,family)
        for k,strategy in enumerate(['直接','多输出'],1):
            doc.add_heading(f'{base+k}｜{family} {strategy}策略：1—7天预测精度',1)
            s=m[(m['模型']==family)&(m['策略']==strategy)]
            assert sorted(s['预见期（天）'].tolist())==list(range(1,8))
            table(s.drop(columns=['模型','策略']))
            add_strategy_analysis(doc,OUT,family,strategy,s)
        image=OUT/(family+'_精度对比.png')
        if not image.exists(): raise FileNotFoundError(image)
        doc.add_picture(str(image),width=Cm(17))
        doc.add_paragraph(f'{family} 两种策略随预见期变化的RMSE与NSE。各指标从同一组测试预测计算。')
    doc.add_heading('ANN训练诊断',1)
    from phase2_diagnostics import collect
    diagnostics=collect(OUT)
    table(diagnostics)
    comparison=pd.read_csv(OUT/'选参方案验证误差对比.csv')
    doc.add_heading('验证误差与训练长度的关系',2)
    table(comparison[['模型','选参预见期','交叉验证RMSE_旧','交叉验证RMSE_新']])
    doc.add_paragraph('扩大计算预算和改变优化器后，不同预见期可能出现改善或退步。此处保留两轮相同时间划分下的验证结果，用于说明训练充分与泛化能力并不等价；模型选择依据交叉验证结果，测试指标用于最终评价。')
    doc.add_paragraph('训练诊断表记录各任务优选模型重拟合后的迭代次数与损失。部分交叉验证候选出现迭代上限或L-BFGS线搜索异常停止警告，详见训练日志。模型效果综合采用交叉验证误差与独立测试指标评价。')
    doc.add_heading('指标解释与结果边界',1)
    doc.add_paragraph('RMSE为均方根误差，MAE为平均绝对误差，单位均为cfs，越小越好。R²与NSE越接近1越好；本实验对相同真实值与预测值使用相同均值参照，两者数值一致，不是两个独立证据。负值表示不如以该测试目标序列均值作常数预测。测试集只用于最终评估，不用于选参；单次年份测试和随机种子结果不等于跨流域泛化。')
    doc.add_paragraph('完整结果见 results/phase2_v2：逐日预测结果.csv、预测精度指标.csv、最佳超参数.csv、各模型网格搜索CSV及交叉验证划分.csv。模型以joblib保存，包含实际参数和训练结果；完整训练日志位于results/phase2_v2_training.log。')
    folder=ROOT/'本地资料/第二阶段'; folder.mkdir(parents=True,exist_ok=True)
    path=folder/'第二阶段报告_ANN与随机森林_正式字体版.docx'; formal_typography(doc); doc.save(path)
    reread=Document(path)
    assert len(reread.inline_shapes)==8 and len(reread.tables)==11
    text='\n'.join(p.text for p in reread.paragraphs)
    assert not any(x in text for x in ['满分','评分标准','全部达标'])
    print(path)

if __name__=='__main__': main()
