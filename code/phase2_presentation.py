"""Reusable explanatory blocks and result figures for phase-two presentation."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from docx.shared import Cm
plt.rcParams['font.sans-serif']=['Microsoft YaHei','SimHei']
plt.rcParams['axes.unicode_minus']=False


def add_method_explanation(doc,family):
    doc.add_heading('模型结构与预测机制',2)
    if family=='ANN':
        doc.add_paragraph('人工神经网络通过多层可训练权重，将输入特征映射为未来径流。隐藏层计算 h=ReLU(Wx+b)，输出层使用线性映射，训练时最小化预测误差与L2惩罚之和。L2惩罚用于限制权重复杂度，降低小样本过拟合风险。缩放输入可避免气温、径流、水汽压等不同量纲妨碍数值优化；缩放目标有助于优化器稳定求解，评价前还原为cfs。')
        doc.add_paragraph('本实验使用L-BFGS拟牛顿优化器，适合中小规模表格数据。它利用目标函数及梯度迭代更新权重，不使用Adam的批次学习率。最大迭代5000与最大函数调用100000是数值保护上限，不是课程要求或必须跑满的训练长度；优化器可在满足停止条件时提前结束。实际迭代次数、损失及警告另行记录，不能用“步数更多”代替泛化效果。')
    else:
        doc.add_paragraph('随机森林通过有放回抽样训练多棵回归树，节点在随机候选特征中选择分裂，再对各树预测取平均。树深度和最小叶节点样本数控制复杂度，树数量影响集成稳定性与计算开销。树模型依据阈值划分输入，对特征线性缩放不敏感，因此使用原始尺度。')
        doc.add_paragraph('单输出树以一个未来目标的误差确定节点划分；多输出树以多个未来目标的联合误差选择共享划分。直接模型允许不同预见期拥有不同结构，多输出模型则共享结构，但可能在不同时间步间折中。')


def add_strategy_analysis(doc,out,family,strategy,metrics):
    preds=pd.read_csv(out/'逐日预测结果.csv')
    sub=preds[(preds['模型']==family)&(preds['策略']==strategy)]
    doc.add_paragraph('直接策略：为每个预见期独立训练与选择一个模型，映射形式为 Q̂(t+h)=f_h(X_t)，h=1,…,7。不同模型不把前一步预测结果递归送回输入，因此没有递归反馈造成的误差传播。' if strategy=='直接' else '多输出策略：一个模型联合输出 [Q̂(t+1),…,Q̂(t+7)]，共享输入和内部结构，利用多个目标间的联系。每个预见期仍单独计算误差，以观察共享结构在近、远期预测上的差异。')
    fig,axes=plt.subplots(2,2,figsize=(11,6))
    for row,h in enumerate([1,7]):
        s=sub[sub['预见期（天）']==h]
        dates=pd.to_datetime(s['目标日期']); y=s['实测径流（cfs）']; p=s['预测径流（cfs）']
        axes[row,0].plot(dates,y,color='#26313b',lw=1,label='实测')
        axes[row,0].plot(dates,p,color='#2f8191',lw=.9,label='预测')
        axes[row,0].set_title(f'第{h}天 · 径流过程'); axes[row,0].legend(fontsize=8)
        axes[row,0].tick_params(axis='x',rotation=20)
        axes[row,1].scatter(y,p,s=8,alpha=.45,color='#2f8191')
        lo=min(y.min(),p.min()); hi=max(y.max(),p.max())
        axes[row,1].plot([lo,hi],[lo,hi],'--',color='#b66b48',lw=1)
        axes[row,1].set_xlabel('实测径流（cfs）'); axes[row,1].set_ylabel('预测径流（cfs）')
        axes[row,1].set_title(f'第{h}天 · 实测与预测散点')
    fig.tight_layout(); path=out/f'{family}_{strategy}_预测过程.png'; fig.savefig(path,dpi=220); plt.close(fig)
    doc.add_picture(str(path),width=Cm(17))
    doc.add_paragraph(f'{family} {strategy}策略第1天与第7天的径流过程及散点。折线以各预见期的目标日期定位，虚线为预测等于实测的参考线；其他预见期完整结果保存在逐日预测CSV。')
    a=metrics.iloc[0]; b=metrics.iloc[-1]
    best=metrics.loc[metrics['RMSE（cfs）'].idxmin()]
    doc.add_paragraph(f"【数值解读】第1天RMSE为{a['RMSE（cfs）']:.2f} cfs、MAE为{a['MAE（cfs）']:.2f} cfs、NSE为{a['NSE']:.4f}；第7天对应RMSE为{b['RMSE（cfs）']:.2f} cfs、NSE为{b['NSE']:.4f}。该策略RMSE最低的预见期为第{int(best['预见期（天）'])}天。误差是否随预见期增大必须以表中结果判断，不能假定严格单调。")
    day1=sub[sub['预见期（天）']==1]
    peak=day1.loc[day1['实测径流（cfs）'].idxmax()]
    error=peak['预测径流（cfs）']-peak['实测径流（cfs）']
    doc.add_paragraph(f"【过程分析】第1天测试实测最大径流为{peak['实测径流（cfs）']:.2f} cfs，当日预测为{peak['预测径流（cfs）']:.2f} cfs，偏差为{error:.2f} cfs。洪峰误差在RMSE中权重较大，需同时观察MAE、散点与洪峰位置，不能只凭单个指标断言模型好坏。")


def add_search_figure(doc,out,family):
    fig,axes=plt.subplots(1,2,figsize=(11,3.5))
    for ax,h in zip(axes,['1','multi']):
        df=pd.read_csv(out/f'{family}_{h}_网格搜索.csv')
        v=-df['mean_test_score']; train=-df['mean_train_score']
        ax.plot(range(1,len(df)+1),v,'o-',label='验证RMSE')
        ax.plot(range(1,len(df)+1),train,'o--',label='训练RMSE')
        ax.set_title('第1天直接模型' if h=='1' else '多输出模型')
        ax.set_xlabel('候选配置编号'); ax.set_ylabel('RMSE（cfs）'); ax.legend(); ax.grid(alpha=.2)
    fig.tight_layout(); p=out/f'{family}_网格搜索对比.png'; fig.savefig(p,dpi=220); plt.close(fig)
    doc.add_picture(str(p),width=Cm(17))
    doc.add_paragraph('候选编号对应网格搜索CSV行顺序。训练误差低而验证误差高，提示泛化差距；参数由验证误差选择，而不以训练误差最低为标准。')
