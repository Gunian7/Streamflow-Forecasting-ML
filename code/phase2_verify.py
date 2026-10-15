from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score
from docx import Document
from report_members import member_fields

ROOT=Path(__file__).resolve().parent.parent
out=ROOT/'results/phase2_v2'
m=pd.read_csv(out/'预测精度指标.csv')
p=pd.read_csv(out/'逐日预测结果.csv')
assert len(m)==28 and len(p)==10052
assert len(list(out.glob('*_网格搜索.csv')))==16
for _,row in m.iterrows():
    s=p[(p['模型']==row['模型'])&(p['策略']==row['策略'])&(p['预见期（天）']==row['预见期（天）'])]
    y=s['实测径流（cfs）']; pred=s['预测径流（cfs）']
    assert len(s)==359
    assert np.isclose(np.sqrt(mean_squared_error(y,pred)),row['RMSE（cfs）'])
    assert np.isclose(mean_absolute_error(y,pred),row['MAE（cfs）'])
    assert np.isclose(r2_score(y,pred),row['R²'])
    assert np.isclose(1-((y-pred)**2).sum()/((y-y.mean())**2).sum(),row['NSE'])
path=ROOT/'本地资料/第二阶段/第二阶段报告_ANN与随机森林_展示修订版.docx'
doc=Document(path)
assert len(doc.inline_shapes)==8 and len(doc.tables)==11
text='\n'.join([p.text for p in doc.paragraphs]+[c.text for t in doc.tables for r in t.rows for c in r.cells])
ids,names=member_fields()
assert all(x in text for x in names.splitlines()+ids.splitlines())
assert all(str(i)+'｜' in text for i in range(1,8))
assert not any(x in text for x in ['一一对应','满分','全部达标','最大迭代350','{var','{{'])
print('28指标、10052预测、16份选参记录、7部分正文、11表、8图验证通过')
print(m[m['预见期（天）']==1].to_string(index=False))
