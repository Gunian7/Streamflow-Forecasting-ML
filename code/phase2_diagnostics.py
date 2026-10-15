from pathlib import Path
import joblib
import pandas as pd

ROOT=Path(__file__).resolve().parent.parent

def collect(out):
    rows=[]
    for p in sorted(out.glob('ANN_*.joblib')):
        search=joblib.load(p)
        model=search.best_estimator_.regressor_.named_steps['model']
        rows.append({'任务':p.stem,'实际迭代次数':model.n_iter_,'最大迭代保护值':model.max_iter,'是否触及迭代上限':model.n_iter_>=model.max_iter,'训练损失（缩放尺度）':model.loss_,'CV_RMSE（cfs）':-search.best_score_})
    frame=pd.DataFrame(rows)
    assert len(frame)==8
    frame.to_csv(out/'ANN训练诊断.csv',index=False,encoding='utf-8-sig')
    old=ROOT/'results/phase2/最佳超参数.csv'
    if old.exists():
        a=pd.read_csv(old); b=pd.read_csv(out/'最佳超参数.csv')
        c=a.merge(b,on=['模型','策略','选参预见期'],suffixes=('_旧','_新'))
        c.to_csv(out/'选参方案验证误差对比.csv',index=False,encoding='utf-8-sig')
    return frame

if __name__=='__main__':
    print(collect(ROOT/'results/phase2_v2').to_string(index=False))
