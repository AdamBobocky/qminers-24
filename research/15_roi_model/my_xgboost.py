import numpy as np
import pandas as pd
import xgboost as xgb

games_df = pd.read_csv('data/games.csv')

games_df['H_ROI'] = games_df['OddsH'] * games_df['H']
games_df['A_ROI'] = games_df['OddsA'] * games_df['A']

# Take the first 20% of dataset, train an xgboost on it that targets H_ROI and A_ROI as regression
# And the inputs are the teams previous seasonal average features

team_stats_average = {}
training_inputs = []
training_outputs = []
model = None
countdown = 4000

oppts = 0
bet_count = 0
pnl = 0

def get_team_four_factor(season, team_id):
    stats = team_stats_average[season][team_id]

    return [
      (stats['FieldGoalsMade'] + 0.5 * stats['3PFieldGoalsMade']) / stats['FieldGoalAttempts'],
      stats['Turnovers'] / (stats['FieldGoalAttempts'] + 0.44 * stats['FreeThrowAttempts'] + stats['Turnovers']),
      stats['OffensiveRebounds'] / (stats['OffensiveRebounds'] + stats['OpponentsDefensiveRebounds']),
      stats['FreeThrowAttempts'] / stats['FieldGoalAttempts']
    ]

for i in games_df.index:
    current = games_df.loc[i]

    season = current['Season']
    home_id = current['HID']
    away_id = current['AID']
    home_win = current['H']
    away_win = current['A']
    odds_home = current['OddsH']
    odds_away = current['OddsA']
    home_roi = current['H_ROI']
    away_roi = current['A_ROI']

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

    if team_stats_average[season][home_id]['Games'] >= 8 and team_stats_average[season][home_id]['Games'] >= 8:
        new_frame_inputs = [*get_team_four_factor(season, home_id), *get_team_four_factor(season, away_id)]
        countdown -= 1
        if countdown == 0:
            countdown = 1000

            np_data_X = np.array(training_inputs)[-8000:]
            np_data_y = np.array(training_outputs)[-8000:]

            model = xgb.XGBRegressor(max_depth=4, n_estimators=60)
            model.fit(np_data_X, np_data_y)

        if model is not None:
            # Make prediction
            exp_roi_home, exp_roi_away = model.predict([new_frame_inputs])[0]

            oppts += 1
            if exp_roi_home > 1.8:
                bet_count += 1
                pnl -= 1
                if home_win:
                    pnl += odds_home

            if exp_roi_away > 1.8:
                bet_count += 1
                pnl -= 1
                if away_win:
                    pnl += odds_away

            if bet_count > 0:
                print('P&L:', pnl / bet_count, bet_count / oppts, bet_count)

        training_inputs.append(new_frame_inputs)
        training_outputs.append([home_roi, away_roi])

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
