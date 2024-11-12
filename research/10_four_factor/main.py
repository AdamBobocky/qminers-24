import math
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression

def inverse_sigmoid(x):
    return math.log(x / (1 - x))

df = pd.read_csv('data/games.csv')

team_stats_average = {}
opponent_stats_average = {}
training_frames = []

corr_me = []
corr_mkt = []
my_mse = 0
mkt_mse = 0
n = 0

def get_team_four_factor(season, team_id):
    stats = team_stats_average[season][team_id]
    opp_stats = opponent_stats_average[season][team_id]

    return [
      (stats['FieldGoalsMade'] + 0.5 * stats['3PFieldGoalsMade']) / stats['FieldGoalAttempts'],
      stats['Turnovers'] / (stats['FieldGoalAttempts'] + 0.44 * stats['FreeThrowAttempts'] + stats['Turnovers']),
      stats['OffensiveRebounds'] / (stats['OffensiveRebounds'] + stats['OpponentsDefensiveRebounds']),
      stats['FreeThrowAttempts'] / stats['FieldGoalAttempts'],
      (opp_stats['FieldGoalsMade'] + 0.5 * opp_stats['3PFieldGoalsMade']) / opp_stats['FieldGoalAttempts'],
      opp_stats['Turnovers'] / (opp_stats['FieldGoalAttempts'] + 0.44 * opp_stats['FreeThrowAttempts'] + opp_stats['Turnovers']),
      opp_stats['OffensiveRebounds'] / (opp_stats['OffensiveRebounds'] + opp_stats['OpponentsDefensiveRebounds']),
      opp_stats['FreeThrowAttempts'] / opp_stats['FieldGoalAttempts']
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
    mkt_pred_factor = inverse_sigmoid(mkt_pred)

    if season not in team_stats_average:
        team_stats_average[season] = {}
    if season not in opponent_stats_average:
        opponent_stats_average[season] = {}
    for team_id in [home_id, away_id]:
        prev_season = season - 1
        if team_id not in team_stats_average[season]:
            if prev_season in team_stats_average and team_id in team_stats_average[prev_season]:
                team_stats_average[season][team_id] = {
                    'FieldGoalsMade': team_stats_average[prev_season][team_id]['FieldGoalsMade'] * 0.2,
                    '3PFieldGoalsMade': team_stats_average[prev_season][team_id]['3PFieldGoalsMade'] * 0.2,
                    'FieldGoalAttempts': team_stats_average[prev_season][team_id]['FieldGoalAttempts'] * 0.2,
                    'Turnovers': team_stats_average[prev_season][team_id]['Turnovers'] * 0.2,
                    'OffensiveRebounds': team_stats_average[prev_season][team_id]['OffensiveRebounds'] * 0.2,
                    'OpponentsDefensiveRebounds': team_stats_average[prev_season][team_id]['OpponentsDefensiveRebounds'] * 0.2,
                    'FreeThrowAttempts': team_stats_average[prev_season][team_id]['FreeThrowAttempts'] * 0.2,
                    'Score': team_stats_average[prev_season][team_id]['Score'] * 0.2,
                    'Games': team_stats_average[prev_season][team_id]['Games'] * 0.2
                }
            else:
                team_stats_average[season][team_id] = {
                    'FieldGoalsMade': 0,
                    '3PFieldGoalsMade': 0,
                    'FieldGoalAttempts': 0,
                    'Turnovers': 0,
                    'OffensiveRebounds': 0,
                    'OpponentsDefensiveRebounds': 0,
                    'FreeThrowAttempts': 0,
                    'Score': 0,
                    'Games': 0
                }
    for team_id in [home_id, away_id]:
        prev_season = season - 1
        if team_id not in opponent_stats_average[season]:
            if prev_season in opponent_stats_average and team_id in opponent_stats_average[prev_season]:
                opponent_stats_average[season][team_id] = {
                    'FieldGoalsMade': opponent_stats_average[prev_season][team_id]['FieldGoalsMade'] * 0.2,
                    '3PFieldGoalsMade': opponent_stats_average[prev_season][team_id]['3PFieldGoalsMade'] * 0.2,
                    'FieldGoalAttempts': opponent_stats_average[prev_season][team_id]['FieldGoalAttempts'] * 0.2,
                    'Turnovers': opponent_stats_average[prev_season][team_id]['Turnovers'] * 0.2,
                    'OffensiveRebounds': opponent_stats_average[prev_season][team_id]['OffensiveRebounds'] * 0.2,
                    'OpponentsDefensiveRebounds': opponent_stats_average[prev_season][team_id]['OpponentsDefensiveRebounds'] * 0.2,
                    'FreeThrowAttempts': opponent_stats_average[prev_season][team_id]['FreeThrowAttempts'] * 0.2,
                    'Score': opponent_stats_average[prev_season][team_id]['Score'] * 0.2,
                    'Games': opponent_stats_average[prev_season][team_id]['Games'] * 0.2
                }
            else:
                opponent_stats_average[season][team_id] = {
                    'FieldGoalsMade': 0,
                    '3PFieldGoalsMade': 0,
                    'FieldGoalAttempts': 0,
                    'Turnovers': 0,
                    'OffensiveRebounds': 0,
                    'OpponentsDefensiveRebounds': 0,
                    'FreeThrowAttempts': 0,
                    'Score': 0,
                    'Games': 0
                }

    if team_stats_average[season][home_id]['Games'] >= 6 and team_stats_average[season][away_id]['Games'] >= 6:
        # new_frame_inputs = [*get_team_four_factor(season, home_id), *get_team_four_factor(season, away_id)]
        new_frame_inputs = [1]
        if len(training_frames) >= 1500:
            np_data = np.array(training_frames)[-2000:]
            X = np_data[:, :-1]
            y = np_data[:, -1]
            lr = LogisticRegression(max_iter=10000)
            lr.fit(X, y)
            # print(X.shape, y.shape)
            print('coefficients:', lr.coef_, 'intercept:', lr.intercept_)
            ensamble_pred = lr.predict_proba(np.array([new_frame_inputs]))[0, 1]
            # print('ensamble_pred', ensamble_pred)


            corr_me.append(ensamble_pred)
            corr_mkt.append(mkt_pred)
            my_mse += (ensamble_pred - home_win) ** 2
            mkt_mse += (mkt_pred - home_win) ** 2
            n += 1

            r = np.corrcoef(corr_me, corr_mkt)[0, 1]
            print('my_mse', my_mse / n, n)
            print('mkt_mse', mkt_mse / n, n)
            print('corr r   ', r)
            print('corr r2  ', r ** 2)

        training_frames.append([*new_frame_inputs, home_win])

    team_stats_average[season][home_id]['FieldGoalsMade'] += current['HFGM']
    team_stats_average[season][home_id]['3PFieldGoalsMade'] += current['HFG3M']
    team_stats_average[season][home_id]['FieldGoalAttempts'] += current['HFGA']
    team_stats_average[season][home_id]['Turnovers'] += current['HTOV']
    team_stats_average[season][home_id]['OffensiveRebounds'] += current['HORB']
    team_stats_average[season][home_id]['OpponentsDefensiveRebounds'] += current['ADRB']
    team_stats_average[season][home_id]['FreeThrowAttempts'] += current['HFTA']
    team_stats_average[season][home_id]['Score'] += current['HSC']
    team_stats_average[season][home_id]['Games'] += 1

    team_stats_average[season][away_id]['FieldGoalsMade'] += current['AFGM']
    team_stats_average[season][away_id]['3PFieldGoalsMade'] += current['AFG3M']
    team_stats_average[season][away_id]['FieldGoalAttempts'] += current['AFGA']
    team_stats_average[season][away_id]['Turnovers'] += current['ATOV']
    team_stats_average[season][away_id]['OffensiveRebounds'] += current['AORB']
    team_stats_average[season][away_id]['OpponentsDefensiveRebounds'] += current['ADRB']
    team_stats_average[season][away_id]['FreeThrowAttempts'] += current['AFTA']
    team_stats_average[season][away_id]['Score'] += current['ASC']
    team_stats_average[season][away_id]['Games'] += 1

    # Opponent
    opponent_stats_average[season][away_id]['FieldGoalsMade'] += current['HFGM']
    opponent_stats_average[season][away_id]['3PFieldGoalsMade'] += current['HFG3M']
    opponent_stats_average[season][away_id]['FieldGoalAttempts'] += current['HFGA']
    opponent_stats_average[season][away_id]['Turnovers'] += current['HTOV']
    opponent_stats_average[season][away_id]['OffensiveRebounds'] += current['HORB']
    opponent_stats_average[season][away_id]['OpponentsDefensiveRebounds'] += current['ADRB']
    opponent_stats_average[season][away_id]['FreeThrowAttempts'] += current['HFTA']
    opponent_stats_average[season][away_id]['Score'] += current['HSC']
    opponent_stats_average[season][away_id]['Games'] += 1

    opponent_stats_average[season][home_id]['FieldGoalsMade'] += current['AFGM']
    opponent_stats_average[season][home_id]['3PFieldGoalsMade'] += current['AFG3M']
    opponent_stats_average[season][home_id]['FieldGoalAttempts'] += current['AFGA']
    opponent_stats_average[season][home_id]['Turnovers'] += current['ATOV']
    opponent_stats_average[season][home_id]['OffensiveRebounds'] += current['AORB']
    opponent_stats_average[season][home_id]['OpponentsDefensiveRebounds'] += current['ADRB']
    opponent_stats_average[season][home_id]['FreeThrowAttempts'] += current['AFTA']
    opponent_stats_average[season][home_id]['Score'] += current['ASC']
    opponent_stats_average[season][home_id]['Games'] += 1
