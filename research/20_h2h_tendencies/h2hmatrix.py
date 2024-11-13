import pandas as pd
import numpy as np

df = pd.read_csv('data/games.csv')

df['Expectation'] = 1 / df['OddsH'] / (1 / df['OddsH'] + 1 / df['OddsA'])

data_df = df[['HID', 'AID', 'OddsH', 'OddsA', 'Expectation', 'H']]

h2hmatrix = {}

pnl = 0
bets = 0

for _, row in data_df.iterrows():
    home_id = row['HID']
    away_id = row['AID']
    home_odds = row['OddsH']
    away_odds = row['OddsA']
    expectation = row['Expectation']
    outcome = row['H']

    key1 = f"{home_id}-{away_id}"
    key2 = f"{away_id}-{home_id}"

    if key1 not in h2hmatrix:
        h2hmatrix[key1] = {
            'exp': 0,
            'truth': 0,
            'n': 0
        }
    if key2 not in h2hmatrix:
        h2hmatrix[key2] = {
            'exp': 0,
            'truth': 0,
            'n': 0
        }

    significance = 1 - 0.985 ** (h2hmatrix[key1]['n'] + h2hmatrix[key2]['n'])

    truth_over_expectation = (h2hmatrix[key1]['truth'] - h2hmatrix[key1]['exp']) - (h2hmatrix[key2]['truth'] - h2hmatrix[key2]['exp'])
    overperformance = truth_over_expectation / (h2hmatrix[key1]['n'] + h2hmatrix[key2]['n'] + 0.001)
    factor = overperformance * significance

    if factor >= 0.06:
        pnl -= 1
        if outcome == 1:
            pnl += home_odds
        bets += 1
    elif factor <= -0.06:
        pnl -= 1
        if outcome == 0:
            pnl += away_odds
        bets += 1

    h2hmatrix[key1]['exp'] += expectation
    h2hmatrix[key1]['truth'] += outcome
    h2hmatrix[key1]['n'] += 1

print(pnl / bets, bets, len(data_df))
