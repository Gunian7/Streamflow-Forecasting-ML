"""Second-stage ANN/RF experiments. Run from any directory."""
import os
os.environ.setdefault('OMP_NUM_THREADS', '1')
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import TransformedTargetRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import step2_feature_engineering as fe

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / 'results/phase2_v2'
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False


def make_data():
    clean = fe.build_features()
    excluded = ['Date','Year','Month','Day','DayOfWeek','DayOfYear','Swe','Discharge']
    cols = [c for c in clean if c not in excluded and not c.startswith('Target_')]
    # All target dates must be strictly before the test year.
    train = clean[clean.Date + pd.Timedelta(days=7) < pd.Timestamp('2004-01-01')]
    test = clean[clean.Date.dt.year == 2004]
    assert not any(c == 'Season' or c == 'Discharge' for c in cols)
    return train, test, cols


def main():
    train, test, cols = make_data()
    X, Xt = train[cols], test[cols]
    ycols = [f'Target_lead{i}' for i in range(1,8)]
    Y, Yt = train[ycols].to_numpy(), test[ycols].to_numpy()
    cv = TimeSeriesSplit(n_splits=3, gap=7)
    folds=[]
    for i,(a,b) in enumerate(cv.split(X),1):
        assert train.iloc[a[-1]].Date + pd.Timedelta(days=7) < train.iloc[b[0]].Date
        folds.append({'折':i,'训练起':str(train.iloc[a[0]].Date.date()),'训练止':str(train.iloc[a[-1]].Date.date()),'验证起':str(train.iloc[b[0]].Date.date()),'验证止':str(train.iloc[b[-1]].Date.date())})
    pd.DataFrame(folds).to_csv(OUT/'交叉验证划分.csv',index=False,encoding='utf-8-sig')
    meta={'训练样本':len(train),'测试样本':len(test),'特征数量':len(cols),'特征':cols,'训练起':str(train.Date.min().date()),'训练止':str(train.Date.max().date()),'测试起':str(test.Date.min().date()),'测试止':str(test.Date.max().date()),'说明':'采用全部非当天径流候选特征，不在全训练集预先进行监督特征筛选。每个模型独立搜索，所有指标为原始cfs量纲。'}
    (OUT/'实验配置.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding='utf-8')
    metrics=[]; best=[]; predictions=[]
    for family in ['ANN','RF']:
        for horizon in list(range(1,8))+['multi']:
            label=f'{family}_{horizon}'
            checkpoint=OUT/(label+'.joblib')
            if family=='ANN':
                pipe=Pipeline([('scale',StandardScaler()),('model',MLPRegressor(solver='lbfgs', max_iter=5000, max_fun=100000, early_stopping=False, random_state=42, tol=1e-5))])
                estimator=TransformedTargetRegressor(regressor=pipe,transformer=StandardScaler())
                grid={'regressor__model__hidden_layer_sizes':[(16,),(32,)],'regressor__model__alpha':[0.1,1.0,10.0],'regressor__model__learning_rate_init':[0.001]}
            else:
                estimator=RandomForestRegressor(n_estimators=300,n_jobs=1,random_state=42)
                grid={'max_depth':[8,16,None],'min_samples_leaf':[1,3,5]}
            target=Y if horizon=='multi' else Y[:,horizon-1]
            if checkpoint.exists():
                search=joblib.load(checkpoint)
            else:
                print('START',label,flush=True)
                search=GridSearchCV(estimator,grid,cv=cv,scoring='neg_root_mean_squared_error',n_jobs=2,error_score='raise',return_train_score=True)
                search.fit(X,target)
                joblib.dump(search,checkpoint,compress=3)
            scores=pd.DataFrame(search.cv_results_)
            scores.to_csv(OUT/(label+'_网格搜索.csv'),index=False,encoding='utf-8-sig')
            best.append({'模型':family,'策略':'多输出' if horizon=='multi' else '直接','选参预见期':str(horizon),'最佳参数':json.dumps(search.best_params_,ensure_ascii=False),'交叉验证RMSE':-search.best_score_})
            pred=search.predict(Xt)
            if horizon!='multi': pred=pred[:,None]
            for j,h in enumerate(range(1,8) if horizon=='multi' else [horizon]):
                actual=Yt[:,h-1]; predicted=pred[:,j]
                rmse=float(np.sqrt(mean_squared_error(actual,predicted)))
                nse=float(1-np.sum((actual-predicted)**2)/np.sum((actual-actual.mean())**2))
                metrics.append({'模型':family,'策略':'多输出' if horizon=='multi' else '直接','预见期（天）':h,'RMSE（cfs）':rmse,'MAE（cfs）':mean_absolute_error(actual,predicted),'R²':r2_score(actual,predicted),'NSE':nse,'样本数':len(actual)})
                for date,a,p in zip(test.Date,actual,predicted):
                    predictions.append({'模型':family,'策略':'多输出' if horizon=='multi' else '直接','起报日期':str(date.date()),'目标日期':str((date+pd.Timedelta(days=h)).date()),'预见期（天）':h,'实测径流（cfs）':a,'预测径流（cfs）':p})
            print('DONE',label,flush=True)
            pd.DataFrame(metrics).to_csv(OUT/'预测精度指标.csv',index=False,encoding='utf-8-sig')
            pd.DataFrame(best).to_csv(OUT/'最佳超参数.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(predictions).to_csv(OUT/'逐日预测结果.csv',index=False,encoding='utf-8-sig')
    m=pd.DataFrame(metrics)
    assert len(m)==28 and len(best)==16
    for family in ['ANN','RF']:
        fig,axes=plt.subplots(1,2,figsize=(11,3.5))
        for strategy in ['直接','多输出']:
            s=m[(m['模型']==family)&(m['策略']==strategy)]
            axes[0].plot(s['预见期（天）'],s['RMSE（cfs）'],'o-',label=strategy)
            axes[1].plot(s['预见期（天）'],s['NSE'],'o-',label=strategy)
        for ax,title in zip(axes,['RMSE（cfs）','NSE']):
            ax.set_title(f'{family} · {title}'); ax.set_xlabel('预见期（天）'); ax.legend(); ax.grid(alpha=.2)
        fig.tight_layout(); fig.savefig(OUT/(family+'_精度对比.png'),dpi=220); plt.close(fig)
    print('COMPLETE',len(m),'metric rows',flush=True)

if __name__=='__main__': main()
