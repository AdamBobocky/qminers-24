import math
import json
import pandas as pd
from datetime import datetime

df = pd.read_csv('data/games.csv')

df['Expectation'] = 1 / df['OddsH'] / (1 / df['OddsH'] + 1 / df['OddsA'])

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def inverse_sigmoid(x):
    return math.log(x / (1 - x))

with open('src/meta_model/data.json', 'r') as file:
    data = json.load(file)

rest_days = 0
last_season = -1
season_start = 0
team_rest = {}
team_games = {}
rest_data = []

for current in data:
    if current['season'] != last_season:
        last_season = current['season']
        season_start = datetime.strptime(current['date'], '%Y-%m-%d %H:%M:%S')

    idx = current['index']
    pred = current['my_pred']
    outcome = current['outcome']
    game = df.loc[int(idx)]
    home_id = game['HID']
    away_id = game['AID']
    # pred = game['Expectation']
    date = datetime.strptime(current['date'], '%Y-%m-%d %H:%M:%S')
    days = (date - season_start).days

    if home_id not in team_games:
        team_games[home_id] = []
    if away_id not in team_games:
        team_games[away_id] = []

    if days >= 20: # and home_id in team_rest and away_id in team_rest
        home_days = (date - team_rest[home_id]).days
        away_days = (date - team_rest[away_id]).days

        if home_days + away_days > 50:
            print(home_days, away_days)
        else:
            adj_pred = inverse_sigmoid(pred)
            # if home_days == 1:  # This is adjustment for when a team played the day before
            #     adj_pred -= 0.2 # This is adjustment for when a team played the day before
            # if away_days == 1:  # This is adjustment for when a team played the day before
            #     adj_pred += 0.2 # This is adjustment for when a team played the day before
            adj_pred = sigmoid(adj_pred)
            rest_data.append([home_days, adj_pred, outcome])
            rest_data.append([away_days, 1 - adj_pred, 1 - outcome])

    team_rest[home_id] = date
    team_rest[away_id] = date

stats = {}

# Populate the dictionary
for key, pred, outcome in rest_data:
    key = min(4, int(key))
    if key not in stats:
        stats[key] = {'wins': 0, 'pred': 0, 'count': 0}
    stats[key]['wins'] += outcome
    stats[key]['pred'] += pred
    stats[key]['count'] += 1

# Calculate the average for each key and store it along with the count
averages = {
    key: {
        'ratio': stats[key]['wins'] / stats[key]['pred'],
        'exp': stats[key]['pred'],
        'wins': stats[key]['wins'],
        'count': stats[key]['count']
    }
    for key in sorted(stats)
}

# Display the result
for key, value in averages.items():
    if value['count'] > 50:
        print(f'Days rest: {key}, expected wins: {value['exp']}, actual wins: {value['wins']}, ratio: {value['ratio'] - 1}, frequency: {value['count'] / len(rest_data)}')
