import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression

df = pd.read_csv('data/games.csv')

team_stats_average = {}
training_frames = []

my_mse = 0
mkt_mse = 0
n = 0

def get_team_four_factor(season, team_id):
    stats = team_stats_average[season][team_id]

    return [
      (stats['FieldGoalsMade'] + 0.5 * stats['3PFieldGoalsMade']) / stats['FieldGoalAttempts'],
      stats['Turnovers'] / (stats['FieldGoalAttempts'] + 0.44 * stats['FreeThrowAttempts'] + stats['Turnovers']),
      stats['OffensiveRebounds'] / (stats['OffensiveRebounds'] + stats['OpponentsDefensiveRebounds']),
      stats['FreeThrowAttempts'] / stats['FieldGoalAttempts']
    ]

for i in df.index:
    current = df.loc[i]

    season = current['Season']
    home_id = current['HID']
    away_id = current['AID']
    home_win = current['H']
    odds_home = current['OddsH']
    odds_away = current['OddsA']
    overround = 1 / odds_home + 1 / odds_away
    mkt_pred = 1 / odds_home / overround

    if season not in team_stats_average:
        team_stats_average[season] = {}
    for team_id in [home_id, away_id]:
        if team_id not in team_stats_average[season]:
            team_stats_average[season][team_id] = {
                'FieldGoalsMade': 0,
                '3PFieldGoalsMade': 0,
                'FieldGoalAttempts': 0,
                'Turnovers': 0,
                'OffensiveRebounds': 0,
                'OpponentsDefensiveRebounds': 0,
                'FreeThrowAttempts': 0,
                'Games': 0
            }

    if team_stats_average[season][home_id]['Games'] >= 6 and team_stats_average[season][home_id]['Games'] >= 6:
        new_frame_inputs = [*get_team_four_factor(season, home_id), *get_team_four_factor(season, away_id)]
        if len(training_frames) >= 500:
            np_data = np.array(training_frames)[-2000:]
            X = np_data[:, :-1]
            y = np_data[:, -1]
            lr = LogisticRegression()
            lr.fit(X, y)
            print('coefficients:', lr.coef_)
            ensamble_pred = lr.predict_proba(np.array(new_frame_inputs).reshape(1, -1))[0, 1]
            print('ensamble_pred', ensamble_pred)

            my_mse += (ensamble_pred - home_win) ** 2
            mkt_mse += (mkt_pred - home_win) ** 2
            n += 1

            print('my_mse', my_mse / n, n)
            print('mkt_mse', mkt_mse / n, n)

        training_frames.append([*new_frame_inputs, home_win])

    team_stats_average[season][home_id]['FieldGoalsMade'] += current['HFGM']
    team_stats_average[season][home_id]['3PFieldGoalsMade'] += current['HFG3M']
    team_stats_average[season][home_id]['FieldGoalAttempts'] += current['HFGA']
    team_stats_average[season][home_id]['Turnovers'] += current['HTOV']
    team_stats_average[season][home_id]['OffensiveRebounds'] += current['HORB']
    team_stats_average[season][home_id]['OpponentsDefensiveRebounds'] += current['ADRB']
    team_stats_average[season][home_id]['FreeThrowAttempts'] += current['HFTA']
    team_stats_average[season][home_id]['Games'] += 1

    team_stats_average[season][away_id]['FieldGoalsMade'] += current['AFGM']
    team_stats_average[season][away_id]['3PFieldGoalsMade'] += current['AFG3M']
    team_stats_average[season][away_id]['FieldGoalAttempts'] += current['AFGA']
    team_stats_average[season][away_id]['Turnovers'] += current['ATOV']
    team_stats_average[season][away_id]['OffensiveRebounds'] += current['AORB']
    team_stats_average[season][away_id]['OpponentsDefensiveRebounds'] += current['ADRB']
    team_stats_average[season][away_id]['FreeThrowAttempts'] += current['AFTA']
    team_stats_average[season][away_id]['Games'] += 1
