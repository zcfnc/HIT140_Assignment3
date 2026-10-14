"""HIT140: reproducible data preparation and classroom linear regression models.

Run: python analysis.py
Raw files are retained unchanged. Outputs are regenerated in data/processed and results.
"""
from pathlib import Path
import csv
import json
import re
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, PowerTransformer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from prematch import TASK1_X, TASK2_X, prepare_corrected

ROOT = Path(__file__).resolve().parent
RAW = ROOT / 'data' / 'raw'
PROCESSED = ROOT / 'data' / 'processed'
RESULTS = ROOT / 'results'
TSDL = 'https://www.thestatsdontlie.com/football/world-cup-2026/'

def prepare_data():
    PROCESSED.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    records = []
    stage = None
    with (RAW / 'tsdl_fixtures.csv').open(encoding='utf-8-sig', newline='') as f:
        for source_row, row in enumerate(csv.reader(f), 1):
            if not row:
                continue
            if row[0] in ['GROUP STAGE', 'ROUND OF 32', 'ROUND OF 16',
                          'QUARTER FINALS', 'SEMI FINALS', 'FINAL']:
                stage = row[0]
            if not re.fullmatch(r'\d{2}/\d{2}/\d{4}', row[0]):
                continue
            assert len(row) == 28, (source_row, len(row))
            score_a, score_b = row[2], row[3]
            assert re.fullmatch(r'p?\d+p?', score_a)
            assert re.fullmatch(r'p?\d+p?', score_b)
            item = dict(date=pd.to_datetime(row[0], format='%d/%m/%Y'), stage=stage,
                        team_a=row[1], team_b=row[4], goals_a=int(score_a.replace('p', '')),
                        goals_b=int(score_b.replace('p', '')), score_a_original=score_a,
                        score_b_original=score_b, source_url=TSDL, source_csv_row=source_row)
            # Source positions verified against the visible two-team table headers.
            for label, offset in [('corners', 17), ('xg', 20), ('shots', 22),
                                  ('sot', 24), ('fouls', 26)]:
                item[f'{label}_a'] = float(row[offset]) if row[offset] else np.nan
                item[f'{label}_b'] = float(row[offset+1]) if row[offset+1] else np.nan
            records.append(item)
    assert len(records) == 103, 'Review any upstream change before merging bronze match'
    extra = json.loads((RAW / 'fifa_bronze_match.json').read_text(encoding='utf-8'))
    extra['date'] = pd.Timestamp(extra['date'])
    records.append(extra)
    matches = pd.DataFrame(records).sort_values(['date', 'team_a', 'team_b']).reset_index(drop=True)
    matches.insert(0, 'match_id', [f'M{i:03}' for i in range(1, len(matches)+1)])
    # These are local analysis IDs, NOT FIFA official match numbers.
    assert len(matches) == 104
    assert not matches.duplicated(['date', 'team_a', 'team_b']).any()
    assert len(set(matches.team_a) | set(matches.team_b)) == 48
    assert matches[['goals_a', 'goals_b']].to_numpy().sum() == 308
    assert (matches.goals_a >= 0).all() and (matches.goals_b >= 0).all()
    assert (matches.team_a != matches.team_b).all()
    assert (matches[matches.stage == 'GROUP STAGE'].shape[0]) == 72
    # Keep calendar dates intact across splits; never split two rows of a match.
    matches['split'] = np.select([matches.date <= '2026-06-25',
                                  matches.date <= '2026-07-01'],
                                 ['train', 'validation'], default='test')
    matches.to_csv(PROCESSED / 'matches_104.csv', index=False)

    return prepare_corrected(matches)

def score(y, pred, train_range):
    mse = mean_squared_error(y, pred)
    return dict(MAE=float(mean_absolute_error(y, pred)), MSE=float(mse),
                RMSE=float(np.sqrt(mse)), NRMSE=float(np.sqrt(mse)/train_range),
                R2=float(r2_score(y, pred)))

def run_task(df, features, target, task):
    out = RESULTS / task
    out.mkdir(parents=True, exist_ok=True)
    train, valid, test = [df[df['split'] == s].copy() for s in ['train', 'validation', 'test']]
    assert len(features) == 8
    assert not set(train.match_id) & set(test.match_id)
    assert not set(valid.match_id) & set(test.match_id)
    assert pd.to_datetime(train.date).max() < pd.to_datetime(valid.date).min()
    assert pd.to_datetime(valid.date).max() < pd.to_datetime(test.date).min()
    assert np.isfinite(df[features]).all().all(), 'Do not impute missing historical data'
    xt, xv, xe = [part[features].to_numpy() for part in [train, valid, test]]
    yt, yv, ye = [part[target].to_numpy() for part in [train, valid, test]]
    target_range = float(np.ptp(yt))
    assert target_range > 0
    pd.DataFrame({'feature':features, 'missing_cells':0, 'imputation_used':False}).to_csv(out/'missingness.csv', index=False)
    train[features+[target]].describe().to_csv(out/'training_descriptive_statistics.csv')
    pd.DataFrame(xt, columns=features).corr().to_csv(out/'training_predictor_correlations.csv')

    # Revised retrospective comparison: old test outcomes have already been seen.
    # No claim of pristine external validation. All candidates keep eight X.
    scaler = StandardScaler().fit(xt)
    power = PowerTransformer(method='yeo-johnson', standardize=True).fit(xt)
    variants = {'raw_ols':(xt,xv,xe),
                'standardised_ols':tuple(scaler.transform(x) for x in [xt,xv,xe]),
                'yeojohnson_ols':tuple(power.transform(x) for x in [xt,xv,xe])}
    assert (xt >= 0).all()
    variants['log1p_ols'] = tuple(np.log1p(x) for x in [xt,xv,xe])
    # Both tasks now use nonnegative team/opponent records. Targets stay raw.
    candidates, validation, models = {}, [], {}
    for name, (x_train,x_valid,x_test) in variants.items():
        model = LinearRegression().fit(x_train, yt)
        models[name] = model
        candidates[name] = (model, x_test)
        validation.append(dict(model=name, **score(yv,model.predict(x_valid),target_range)))
    validation.append(dict(model='training_mean_baseline', **score(yv,np.repeat(yt.mean(),len(yv)),target_range)))
    vtable = pd.DataFrame(validation)
    vtable.to_csv(out/'validation_comparison.csv', index=False)
    eligible = vtable[vtable.model != 'training_mean_baseline']
    winner = eligible.sort_values(['RMSE','model']).iloc[0].model
    # Deterministic tie convention: raw and scaled OLS represent the same model.
    if np.isclose(float(eligible[eligible.model=='raw_ols'].RMSE.iloc[0]),eligible.RMSE.min(),atol=1e-10):
        winner = 'raw_ols'
    chosen, test_x = candidates[winner]
    pred = chosen.predict(test_x)
    base = np.repeat(yt.mean(),len(ye))
    final = pd.DataFrame([dict(model=winner,**score(ye,pred,target_range)),
                          dict(model='training_mean_baseline',**score(ye,base,target_range))])
    final.to_csv(out/'test_metrics.csv', index=False)
    # Supplementary classroom comparison: one raw predictor versus eight raw predictors.
    # Fix the single predictor by its meaning, without selecting it on test performance.
    simple_feature = 'team_pre_gf' if task == 'task22' else 'a_pre_gf'
    simple_index = features.index(simple_feature)
    simple = LinearRegression().fit(xt[:, [simple_index]], yt)
    simple_pred = simple.predict(xe[:, [simple_index]])
    multiple_pred = models['raw_ols'].predict(xe)
    comparison = pd.DataFrame([
        dict(model='simple_raw_ols', predictor_count=1, **score(ye, simple_pred, target_range)),
        dict(model='multiple_raw_ols', predictor_count=8, **score(ye, multiple_pred, target_range))])
    comparison.to_csv(out/'simple_vs_multiple_metrics.csv', index=False)
    paired = test[['match_id', 'date'] + (['team'] if 'team' in test else ['team_a', 'team_b'])].copy()
    paired['actual'] = ye
    paired['simple_prediction'] = simple_pred
    paired['multiple_prediction'] = multiple_pred
    paired.to_csv(out/'simple_vs_multiple_predictions.csv', index=False)
    details = dict(simple_feature=simple_feature,
        feature_meaning='Mean goals scored in the previous five completed internationals',
        simple_intercept=float(simple.intercept_), simple_coefficient=float(simple.coef_[0]),
        multiple_features=features, multiple_intercept=float(models['raw_ols'].intercept_),
        multiple_coefficients=models['raw_ols'].coef_.tolist(),
        training_rows=len(train), test_rows=len(test), nrmse_denominator=target_range,
        note='Same training and test rows; raw predictors and response; OLS with intercept. '
             'Supplementary model comparison, not different algorithms. The required eight-predictor '
             'primary analysis remains separate. The later evaluation period is retrospective.')
    (out/'simple_vs_multiple_details.json').write_text(json.dumps(details, indent=2), encoding='utf-8')
    recommendation = vtable.sort_values(['RMSE','model']).iloc[0].model
    prediction = test[['match_id','date']+(['team'] if 'team' in test else ['team_a','team_b'])].copy()
    prediction['actual'] = ye
    prediction['predicted'] = pred
    prediction['baseline'] = base
    prediction['residual'] = ye-pred
    prediction.to_csv(out/'test_predictions.csv', index=False)
    coefficient = pd.DataFrame({'feature':features,'coefficient':chosen.coef_})
    coefficient.to_csv(out/'selected_coefficients.csv', index=False)
    # Statsmodels is a second implementation of OLS, NOT a new algorithm.
    selected_train = variants[winner][0]
    smmodel = sm.OLS(yt,sm.add_constant(pd.DataFrame(selected_train,columns=features),has_constant='add')).fit()
    assert np.allclose(smmodel.predict(sm.add_constant(selected_train,has_constant='add')),chosen.predict(selected_train))
    (out/'statsmodels_training_summary.txt').write_text(smmodel.summary().as_text(), encoding='utf-8')
    equation = f'Predicted {target} = {chosen.intercept_:.6f}'
    equation += ''.join(f' {c:+.6f} * {f}' for f,c in zip(features, chosen.coef_))
    if winner == 'yeojohnson_ols':
        equation += '\nFeatures in this equation are training-fitted Yeo-Johnson transformed and standardised values, not raw values.'
    if winner == 'log1p_ols':
        equation += '\nFeatures in this equation are log(1 + raw historical feature).'
    (out/'selected_equation.txt').write_text(equation,encoding='utf-8')
    # T8 correlation exercise: drop later columns if |r|>0.8 on TRAIN only.
    corr = np.corrcoef(xt,rowvar=False)
    keep = []
    for j in range(len(features)):
        if not any(abs(corr[j,k]) > .8 for k in keep):
            keep.append(j)
    reduced = LinearRegression().fit(xt[:,keep],yt)
    (out/'t8_reduction_exercise.json').write_text(json.dumps(dict(
        retained_features=[features[j] for j in keep],
        validation=score(yv,reduced.predict(xv[:,keep]),target_range),
        note='Classroom exercise only. Not eligible for final selection because the brief requires eight predictors.'),indent=2),encoding='utf-8')
    summary = dict(task=task, selected_model=winner, selection='Lowest validation RMSE; raw OLS wins numerical ties.',
                   predictor_count=8, rows={s:int((df['split']==s).sum()) for s in ['train','validation','test']},
                   intercept=float(chosen.intercept_), nrmse_denominator=target_range,
                   training_adjusted_r2=float(smmodel.rsquared_adj),
                   validation_recommendation=recommendation,
                   evaluation_status='Retrospective revised comparison of a later tournament period.',
                   test=final.to_dict(orient='records'),
                   negative_test_predictions=int((pred<0).sum()),
                   cold_start_rows=int(df[features].isna().all(axis=1).sum()))
    (out/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    # Training-only plots for exploration; holdout plots appear after selection.
    fig, axes = plt.subplots(2,4,figsize=(14,6))
    for ax,f in zip(axes.flat,features):
        ax.scatter(train[f],yt,s=15,alpha=.6)
        ax.set_xlabel(f,fontsize=8); ax.set_ylabel(target,fontsize=8)
    fig.suptitle(f'{task}: training predictors and response')
    fig.tight_layout(); fig.savefig(out/'training_scatterplots.png',dpi=160); plt.close(fig)
    fig,ax=plt.subplots(figsize=(9,8))
    im=ax.imshow(corr,vmin=-1,vmax=1,cmap='coolwarm')
    ax.set_xticks(range(8),features,rotation=60,ha='right',fontsize=8)
    ax.set_yticks(range(8),features,fontsize=8)
    fig.colorbar(im,ax=ax,label='Pearson correlation (training data)')
    fig.tight_layout(); fig.savefig(out/'training_correlation.png',dpi=160); plt.close(fig)
    fig,axs=plt.subplots(1,2,figsize=(10,4))
    axs[0].scatter(ye,pred,alpha=.7)
    lo,hi=min(ye.min(),pred.min()),max(ye.max(),pred.max())
    axs[0].plot([lo,hi],[lo,hi],'k--',lw=1)
    axs[0].set(xlabel='Actual goals / goal difference',ylabel='Predicted',title='Later-period evaluation (reused)')
    axs[1].scatter(pred,ye-pred,alpha=.7); axs[1].axhline(0,color='black',ls='--')
    axs[1].set(xlabel='Predicted',ylabel='Actual minus predicted',title='Test residuals')
    fig.suptitle(task+' — '+winner); fig.tight_layout()
    fig.savefig(out/'test_diagnostics.png',dpi=160); plt.close(fig)
    errors = prediction.assign(model_absolute_error=np.abs(ye-pred),
                               baseline_absolute_error=np.abs(ye-base))
    grouped = errors.groupby('match_id')[['model_absolute_error','baseline_absolute_error']].mean()
    grouped['improvement'] = grouped.baseline_absolute_error-grouped.model_absolute_error
    grouped.to_csv(out/'match_error_comparison.csv')
    ordered = grouped.sort_values('improvement')
    fig,ax=plt.subplots(figsize=(10,6))
    ax.barh(ordered.index,ordered.improvement,color=np.where(ordered.improvement>=0,'#197a82','#c16a33'))
    ax.axvline(0,color='#263548',linewidth=1)
    ax.set(xlabel='Baseline absolute error minus regression absolute error (goals)',
           title=f'{task}: where does regression help?\nPositive = improvement; later period reused')
    ax.tick_params(axis='y',labelsize=8)
    fig.tight_layout();fig.savefig(out/'match_error_story.png',dpi=160);plt.close(fig)
    fig,ax=plt.subplots(figsize=(9,4))
    ax.barh(vtable.model,vtable.RMSE,color=['#2c6c9b']*(len(vtable)-1)+['#8b8b8b'])
    ax.set(xlabel='Validation RMSE (goals; lower is better)',title=f'{task}: OLS variants versus baseline')
    fig.tight_layout();fig.savefig(out/'validation_comparison.png',dpi=160);plt.close(fig)
    return summary

def run_all():
    one,two=prepare_data()
    summaries=[run_task(one,TASK1_X,'goal_difference','task21'),
               run_task(two,TASK2_X,'goals','task22')]
    print(json.dumps(summaries,indent=2))
    return summaries

if __name__ == '__main__':
    run_all()
