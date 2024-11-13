import math
import json
import pandas as pd
import numpy as np
import copy
from collections import defaultdict

players_df = pd.read_csv('data/players.csv')
games_df = pd.read_csv('data/games.csv')

total_games = len(games_df)

team_rosters = {}
team_trigger = {}

trigger_exp = 0
trigger_truth = 0
trigger_n = 0

for index, current in games_df.iterrows():
    season = current['Season']
    home_id = current['HID']
    away_id = current['AID']
    home_win = current['H']
    away_win = current['A']
    home_score = current['HSC']
    away_score = current['ASC']
    odds_home = current['OddsH']
    odds_away = current['OddsA']
    overround = 1 / odds_home + 1 / odds_away
    mkt_pred = 1 / odds_home / overround

    if home_id in team_trigger and team_trigger[home_id] == True and (away_id not in team_trigger or team_trigger[away_id] == False):
        trigger_exp += mkt_pred
        trigger_truth += home_win
        trigger_n += 1

        del team_trigger[home_id]

    if away_id in team_trigger and team_trigger[away_id] == True and (home_id not in team_trigger or team_trigger[home_id] == False):
        trigger_exp += 1 - mkt_pred
        trigger_truth += away_win
        trigger_n += 1

        del team_trigger[away_id]

    # Make prediction
    if season in team_rosters and home_id in team_rosters[season] and away_id in team_rosters[season] and len(team_rosters[season][home_id]) >= 6 and len(team_rosters[season][away_id]) >= 6:
        home_rosters = team_rosters[season][home_id][-12:-1]
        away_rosters = team_rosters[season][away_id][-12:-1]
        last_home_roster = team_rosters[season][home_id][-1]
        last_away_roster = team_rosters[season][away_id][-1]

        home_roster = defaultdict(int)
        away_roster = defaultdict(int)

        for roster in home_rosters:
            for pid, mins in roster:
                home_roster[pid] += mins

        for roster in away_rosters:
            for pid, mins in roster:
                away_roster[pid] += mins

        home_roster = [x[0] for x in sorted(home_roster.items(), key=lambda x: x[1], reverse=True)[:5]]
        away_roster = [x[0] for x in sorted(away_roster.items(), key=lambda x: x[1], reverse=True)[:5]]
        last_home_roster = [x[0] for x in sorted(last_home_roster, key=lambda x: x[1], reverse=True)[:8]]
        last_away_roster = [x[0] for x in sorted(last_away_roster, key=lambda x: x[1], reverse=True)[:8]]

        home_missing = [item for item in home_roster if item not in last_home_roster]
        away_missing = [item for item in away_roster if item not in last_away_roster]

        team_trigger[home_id] = len(home_missing) >= 1
        team_trigger[away_id] = len(away_missing) >= 1

    game_players = players_df[(players_df['Game'] == index) & (players_df['MIN'] >= 3)]

    home_players = game_players[game_players['Team'] == current['HID']]
    away_players = game_players[game_players['Team'] == current['AID']]

    if season not in team_rosters:
        team_rosters[season] = {}

    if home_id not in team_rosters[season]:
        team_rosters[season][home_id] = []

    if away_id not in team_rosters[season]:
        team_rosters[season][away_id] = []

    team_rosters[season][home_id].append([[x['Player'], x['MIN']] for _, x in home_players.iterrows()])
    team_rosters[season][away_id].append([[x['Player'], x['MIN']] for _, x in away_players.iterrows()])

    if (index + 1) % (total_games // 1000) == 0:
        progress = (index + 1) / total_games * 100
        print(f'Progress: {progress:.1f}%')

print(trigger_exp, trigger_truth, trigger_n)

print('Done')
