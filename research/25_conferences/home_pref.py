import json
import pandas as pd
import numpy as np

with open('src/meta_model/data.json', 'r') as f:
    data = json.load(f)

pred_map = {}

for point in data:
    pred_map[int(point['index'])] = point['my_pred']

df = pd.read_csv('data/games.csv')

df['Expectation'] = 1 / df['OddsH'] / (1 / df['OddsH'] + 1 / df['OddsA'])

data_df = df[['HID', 'AID', 'OddsH', 'OddsA', 'Expectation', 'H']]

home_pref = {}

expected = 0
actual = 0
n = 0

pnl = 0
bets = 0

for idx, row in data_df.iterrows():
    home_id = row['HID']
    away_id = row['AID']
    # home_odds = row['OddsH']
    # away_odds = row['OddsA']
    expectation = row['Expectation']
    outcome = row['H']

    if home_id not in home_pref:
        home_pref[home_id] = {
            'exp': 0,
            'truth': 0,
            'n': 0
        }

    significance = 1 - 0.985 ** home_pref[home_id]['n']

    truth_over_expectation = home_pref[home_id]['truth'] - home_pref[home_id]['exp']
    overperformance = truth_over_expectation / (home_pref[home_id]['n'] + 0.001)
    factor = overperformance * significance

    if idx in pred_map:
        my_pred = pred_map[idx]

        if factor >= 0.02:
            pnl -= 1
            if outcome == 1:
                pnl += 1 / my_pred
            bets += 1
            expected += my_pred
            actual += outcome
            n += 1
        elif factor <= -0.02:
            pnl -= 1
            if outcome == 0:
                pnl += 1 / (1 - my_pred)
            bets += 1
            expected += 1 - my_pred
            actual += 1 - outcome
            n += 1

    home_pref[home_id]['exp'] += expectation
    home_pref[home_id]['truth'] += outcome
    home_pref[home_id]['n'] += 1

print(pnl / bets, bets, len(data_df))
print(expected / actual, expected, actual, n)
