"""Auditable last-five completed internationals. No cold-start imputation."""
from pathlib import Path
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
METRICS = ['gf', 'ga', 'win_rate', 'clean_sheet_rate']
ALIASES = {'United States': 'USA', 'Curaçao': 'Curacao',
           'DR Congo': 'D.R. Congo', 'Bosnia and Herzegovina': 'Bosnia & Herzegovina'}
TASK1_X = [f'{side}_pre_{metric}' for side in ['a', 'b'] for metric in METRICS]
TASK2_X = [f'{side}_pre_{metric}' for side in ['team', 'opponent'] for metric in METRICS]

def load_history():
    frame = pd.read_csv(ROOT/'data/raw/international_results.csv', parse_dates=['date'])
    frame.insert(0, 'source_csv_row', np.arange(2, len(frame)+2))
    for col in ['home_team', 'away_team']:
        frame[col] = frame[col].replace(ALIASES)
    return frame

def reconcile(matches, history):
    """Independent source corroboration, without using its outcomes as predictors."""
    wc = history[(history.tournament == 'FIFA World Cup') &
                 (history.date.between('2026-06-11', '2026-07-19'))]
    assert len(wc) == 104
    rows = []
    for m in matches.itertuples():
        candidate = wc[((wc.home_team == m.team_a) & (wc.away_team == m.team_b)) |
                       ((wc.home_team == m.team_b) & (wc.away_team == m.team_a))]
        candidate = candidate[(candidate.date-m.date).abs() <= pd.Timedelta(days=1)]
        assert len(candidate) == 1, (m.match_id, m.team_a, m.team_b)
        r = candidate.iloc[0]
        ga, gb = (r.home_score, r.away_score) if r.home_team == m.team_a else (r.away_score, r.home_score)
        assert (m.goals_a, m.goals_b) == (ga, gb), ('Score conflict', m.match_id)
        rows.append(dict(match_id=m.match_id, fixture_date=m.date,
            corroborating_date=r.date, date_offset_days=(r.date-m.date).days,
            scores_agree=True, history_source_row=int(r.source_csv_row)))
    return pd.DataFrame(rows)

def build_features(matches, history, window=5):
    """History source only before tournament; within-tournament history from fixtures.

    A historical snapshot downloaded after the event is a retrospective proxy,
    not proof that this particular CSV version was published before kickoff.
    """
    historical = history[(history.date >= '2024-01-01') & (history.date < '2026-06-11')]
    past = []
    for r in historical.itertuples():
        for team, opp, gf, ga in [(r.home_team,r.away_team,r.home_score,r.away_score),
                                  (r.away_team,r.home_team,r.away_score,r.home_score)]:
            past.append(dict(team=team, opponent=opp, date=r.date, gf=gf, ga=ga,
                source='international_results.csv', source_row=r.source_csv_row,
                history_id=f'H{r.source_csv_row}', source_match_id=''))
    for r in matches.itertuples():
        for team, opp, gf, ga in [(r.team_a,r.team_b,r.goals_a,r.goals_b),
                                  (r.team_b,r.team_a,r.goals_b,r.goals_a)]:
            past.append(dict(team=team, opponent=opp, date=r.date, gf=gf, ga=ga,
                source='tsdl_fixtures.csv' if r.stage != 'THIRD PLACE' else 'fifa_bronze_match.json',
                source_row=r.source_csv_row, history_id=r.match_id, source_match_id=r.match_id))
    pool = pd.DataFrame(past)
    assert not pool.duplicated(['team', 'opponent', 'date']).any()
    team_rows, links = [], []
    for m in matches.itertuples():
        for side, team, opponent, gf, ga in [('a',m.team_a,m.team_b,m.goals_a,m.goals_b),
                                           ('b',m.team_b,m.team_a,m.goals_b,m.goals_a)]:
            earlier = pool[(pool.team == team) & (pool.date < m.date)].sort_values(['date','history_id'])
            recent = earlier.tail(window)
            assert len(recent) == window, ('Insufficient genuine history', team, m.date)
            row = dict(match_id=m.match_id, date=m.date, stage=m.stage, split=m.split,
                team=team, opponent=opponent, side=side, gf=gf, ga=ga,
                prior_matches=len(earlier), history_count=window,
                history_first_date=recent.date.min(), history_last_date=recent.date.max(),
                pre_gf=recent.gf.mean(), pre_ga=recent.ga.mean(),
                pre_win_rate=(recent.gf > recent.ga).mean(),
                pre_clean_sheet_rate=(recent.ga == 0).mean())
            team_rows.append(row)
            for rank, r in enumerate(recent.itertuples(),1):
                links.append(dict(match_id=m.match_id, team=team, forecast_date=m.date,
                    history_order=rank, history_date=r.date, history_opponent=r.opponent,
                    gf=r.gf, ga=r.ga, source=r.source, source_row=r.source_row,
                    history_id=r.history_id))
    teams = pd.DataFrame(team_rows)
    one, two = [], []
    for m in matches.itertuples():
        pair = teams[teams.match_id == m.match_id].set_index('side')
        a,b = pair.loc['a'],pair.loc['b']
        row = dict(match_id=m.match_id, date=m.date, split=m.split,
                   team_a=m.team_a, team_b=m.team_b, goal_difference=m.goals_a-m.goals_b)
        for side, entry in [('a',a),('b',b)]:
            row.update({f'{side}_pre_{k}':entry[f'pre_{k}'] for k in METRICS})
        one.append(row)
        for own,opp in [(a,b),(b,a)]:
            row = dict(match_id=m.match_id, date=m.date, split=m.split,
                       team=own.team, opponent=opp.team, goals=own.gf)
            for side,entry in [('team',own),('opponent',opp)]:
                row.update({f'{side}_pre_{k}':entry[f'pre_{k}'] for k in METRICS})
            two.append(row)
    one,two = pd.DataFrame(one),pd.DataFrame(two)
    assert np.isfinite(one[TASK1_X]).all().all() and np.isfinite(two[TASK2_X]).all().all()
    return one,two,teams,pd.DataFrame(links)

def prepare_corrected(matches):
    history = load_history()
    agreement = reconcile(matches, history)
    one,two,teams,links = build_features(matches,history)
    out = ROOT/'data/processed'
    for name,frame in [('task21_match_features',one),('task22_team_features',two),
                       ('team_match_208_audit',teams),('feature_history_lineage',links),
                       ('source_reconciliation_104',agreement)]:
        frame.to_csv(out/f'{name}.csv',index=False)
    audit = dict(matches=len(matches),team_rows=len(teams),unique_teams=teams.team.nunique(),
        total_goals=int(teams.gf.sum()),history_window=5,missing_predictor_cells=0,
        imputation_used=False,history_strictly_earlier=bool((links.history_date<links.forecast_date).all()),
        independently_corroborated_scores=len(agreement),
        date_offset_counts=agreement.date_offset_days.value_counts().to_dict(),
        matches_by_split=matches['split'].value_counts().to_dict(),
        history_caution='Retrospective snapshot, not an archived pre-kickoff release.',
        evaluation_caution='Later-period evaluation is retrospective and is not treated as untouched external validation.')
    (ROOT/'results/data_audit.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
    return one,two
