import math
import json
import pandas as pd
from datetime import datetime

df = pd.read_csv('data/games.csv')

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def inverse_sigmoid(x):
    return math.log(x / (1 - x))

with open('src/meta_model/data.json', 'r') as file:
    data = json.load(file)

team_games = {}

momentum_map = {
    0: [0, 0, 0],
    1: [0, 0, 0],
    2: [0, 0, 0],
    3: [0, 0, 0],
    4: [0, 0, 0],
    5: [0, 0, 0]
}

for current in data:
    idx = current['index']
    pred = current['mkt_pred']
    outcome = current['outcome']
    game = df.loc[int(idx)]
    season = game['Season']
    home_id = game['HID']
    away_id = game['AID']

    if season not in team_games:
        team_games[season] = {}
    if home_id not in team_games[season]:
        team_games[season][home_id] = []
    if away_id not in team_games[season]:
        team_games[season][away_id] = []

    if len(team_games[season][home_id]) >= 5 and len(team_games[season][away_id]) >= 5:
        home_last_5 = team_games[season][home_id][-5:]
        away_last_5 = team_games[season][away_id][-5:]

        home_key = sum(home_last_5)
        away_key = sum(away_last_5)

        momentum_map[home_key][0] += pred
        momentum_map[home_key][1] += outcome
        momentum_map[home_key][2] = momentum_map[home_key][1] / momentum_map[home_key][0] - 1

        momentum_map[away_key][0] += 1 - pred
        momentum_map[away_key][1] += 1 - outcome
        momentum_map[away_key][2] = momentum_map[away_key][1] / momentum_map[away_key][0] - 1

    team_games[season][home_id].append(outcome)
    team_games[season][away_id].append(1 - outcome)

print(momentum_map)
