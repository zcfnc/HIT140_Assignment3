"""Read-only integrity and leakage tests of the regenerated submission analysis."""
from pathlib import Path
import hashlib, json, re
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from prematch import TASK1_X,TASK2_X,build_features,load_history
ROOT=Path(__file__).resolve().parent
def verify():
    p=ROOT/'data/processed'
    m=pd.read_csv(p/'matches_104.csv',parse_dates=['date'])
    a=pd.read_csv(p/'task21_match_features.csv',parse_dates=['date'])
    b=pd.read_csv(p/'task22_team_features.csv',parse_dates=['date'])
    t=pd.read_csv(p/'team_match_208_audit.csv',parse_dates=['date'])
    links=pd.read_csv(p/'feature_history_lineage.csv',parse_dates=['forecast_date','history_date'])
    assert len(m)==len(a)==104 and len(t)==len(b)==208
    assert len(TASK1_X)==len(TASK2_X)==8
    assert np.isfinite(a[TASK1_X]).all().all() and np.isfinite(b[TASK2_X]).all().all()
    assert t.team.nunique()==48 and t.gf.sum()==308
    assert not m.duplicated(['date','team_a','team_b']).any()
    assert np.array_equal(a.goal_difference,m.goals_a-m.goals_b)
    for manifest in ['sources.json','history_sources.json']:
        for s in json.loads((ROOT/'data/raw'/manifest).read_text()):
            assert hashlib.sha256((ROOT/s['file']).read_bytes()).hexdigest()==s['sha256']
    assert len(links)==208*5 and (links.history_date<links.forecast_date).all()
    for r in t.itertuples():
        h=links[(links.match_id==r.match_id)&(links.team==r.team)]
        assert len(h)==5
        assert np.isclose(r.pre_gf,h.gf.mean()) and np.isclose(r.pre_ga,h.ga.mean())
        assert np.isclose(r.pre_win_rate,(h.gf>h.ga).mean())
        assert np.isclose(r.pre_clean_sheet_rate,(h.ga==0).mean())
    for _,pair in t.groupby('match_id'):
        assert len(pair)==2 and pair['split'].nunique()==1
        x,y=pair.iloc[0],pair.iloc[1]
        assert x.gf==y.ga and x.ga==y.gf and x.team==y.opponent
    # Recalculate directly from raw history and compare every exported predictor.
    hist=load_history()
    ra,rb,_,_=build_features(m,hist)
    assert np.allclose(ra[TASK1_X],a[TASK1_X])
    assert np.allclose(rb[TASK2_X],b[TASK2_X])
    # Changing current/future outcomes must not change current/past predictors.
    altered=m.copy()
    cutoff=pd.Timestamp('2026-07-01')
    altered.loc[altered.date>=cutoff,['goals_a','goals_b']]+=99
    ca,cb,_,_=build_features(altered,hist)
    assert np.allclose(ca.loc[ca.date<=cutoff,TASK1_X],a.loc[a.date<=cutoff,TASK1_X])
    assert np.allclose(cb.loc[cb.date<=cutoff,TASK2_X],b.loc[b.date<=cutoff,TASK2_X])
    # Ignore all post-start rows in the external historical snapshot.
    changed=hist.copy()
    changed.loc[changed.date>='2026-06-11',['home_score','away_score']]+=99
    da,db,_,_=build_features(m,changed)
    assert np.allclose(da[TASK1_X],a[TASK1_X]) and np.allclose(db[TASK2_X],b[TASK2_X])
    for df,features,target,task in [(a,TASK1_X,'goal_difference','task21'),(b,TASK2_X,'goals','task22')]:
        assert df.groupby('match_id')['split'].nunique().eq(1).all()
        tr,va,te=[df[df['split']==s] for s in ['train','validation','test']]
        assert tr.date.max()<va.date.min() and va.date.max()<te.date.min()
        out=ROOT/'results'/task
        predictions=pd.read_csv(out/'test_predictions.csv')
        metrics=pd.read_csv(out/'test_metrics.csv')
        for i,col in enumerate(['predicted','baseline']):
            r=metrics.iloc[i]
            assert np.isclose(r.MAE,mean_absolute_error(predictions.actual,predictions[col]))
            assert np.isclose(r.MSE,mean_squared_error(predictions.actual,predictions[col]))
            assert np.isclose(r.RMSE,np.sqrt(r.MSE))
            assert np.isclose(r.NRMSE,r.RMSE/np.ptp(tr[target]))
            assert np.isclose(r.R2,r2_score(predictions.actual,predictions[col]))
        comparison=pd.read_csv(out/'validation_comparison.csv').set_index('model')
        assert np.isclose(comparison.loc['raw_ols','RMSE'],comparison.loc['standardised_ols','RMSE'])
        summary=json.loads((out/'summary.json').read_text())
        assert np.isclose(comparison.loc[summary['selected_model'],'RMSE'],comparison.drop('training_mean_baseline').RMSE.min())
        assert np.allclose(predictions.baseline,tr[target].mean())
        assert not (out/'imputation.csv').exists()
        paired=pd.read_csv(out/'simple_vs_multiple_predictions.csv')
        comparison2=pd.read_csv(out/'simple_vs_multiple_metrics.csv')
        assert paired.match_id.tolist()==te.match_id.tolist()
        assert np.allclose(paired.actual,te[target])
        for i,cols in enumerate([[features[0]],features]):
            fitted=LinearRegression().fit(tr[cols],tr[target])
            predicted=fitted.predict(te[cols])
            assert np.allclose(predicted,paired[['simple_prediction','multiple_prediction'][i]])
            row=comparison2.iloc[i]
            assert row.predictor_count==len(cols)
            assert np.isclose(row.MAE,mean_absolute_error(te[target],predicted))
            assert np.isclose(row.RMSE,np.sqrt(mean_squared_error(te[target],predicted)))
            assert np.isclose(row.NRMSE,row.RMSE/np.ptp(tr[target]))
    cells=0
    for file in ['00_Data_Preparation.ipynb','01_Task21_Goal_Difference.ipynb','02_Task22_Team_Goals.ipynb']:
        nb=json.loads((ROOT/file).read_text(encoding='utf-8'))
        assert nb['nbformat']==4
        ids=[c['id'] for c in nb['cells']]
        assert len(ids)==len(set(ids))
        for c in nb['cells']:
            if c['cell_type']=='code':
                compile(''.join(c['source']),file,'exec')
                assert c['execution_count'] and all(o['output_type']!='error' for o in c['outputs'])
                cells+=1
            else:
                for image in re.findall(r'!\[[^\]]*\]\(([^)]+)\)',''.join(c['source'])):
                    assert (ROOT/image).exists(),image
    print('PASS: counts, 8 X each, finite features, raw hashes, all 1040 lineage rows,')
    print('raw-data recalculation, pairing, chronological splits, future-outcome perturbation,')
    print('external future-data exclusion, both models/baselines metrics, model selection, image links.')
    print(f'PASS: {cells} executed plain-Python notebook cells; JSON structure and syntax.')
    print('Scope: code, data integrity, leakage, metrics and notebook structure checks.')
if __name__=='__main__': verify()
