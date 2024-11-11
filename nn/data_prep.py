import math
import json
import pandas as pd
import numpy as np
import copy
from collections import defaultdict
from datetime import datetime

players_df = pd.read_csv('data/players.csv')
games_df = pd.read_csv('data/games.csv')

keys = []
player_data = defaultdict(list)
home_inputs = []
away_inputs = []
home_playtimes = []
away_playtimes = []
outputs = []
team_rosters = {}

class FourFactor:
    def __init__(self):
        self.team_stats_average = defaultdict(list)
        self.opponent_stats_average = defaultdict(list)
        self.cache = {}

    def _get_stats(self, date, stats):
        totals = {
            'FieldGoalsMade': 0,
            '3PFieldGoalsMade': 0,
            'FieldGoalAttempts': 0,
            'Turnovers': 0,
            'OffensiveRebounds': 0,
            'OpponentsDefensiveRebounds': 0,
            'FreeThrowAttempts': 0,
            'Score': 0,
            'Win': 0,
            'Weight': 0
        }

        date = datetime.strptime(date, '%Y-%m-%d')

        # Iterate over each dictionary in the list
        for stat in stats:
            weight = 0.994 ** abs((date - datetime.strptime(stat['Date'], '%Y-%m-%d')).days)

            # Multiply each relevant field by the weight and add to totals
            totals['FieldGoalsMade'] += stat['FieldGoalsMade'] * weight
            totals['3PFieldGoalsMade'] += stat['3PFieldGoalsMade'] * weight
            totals['FieldGoalAttempts'] += stat['FieldGoalAttempts'] * weight
            totals['Turnovers'] += stat['Turnovers'] * weight
            totals['OffensiveRebounds'] += stat['OffensiveRebounds'] * weight
            totals['OpponentsDefensiveRebounds'] += stat['OpponentsDefensiveRebounds'] * weight
            totals['FreeThrowAttempts'] += stat['FreeThrowAttempts'] * weight
            totals['Score'] += stat['Score'] * weight
            totals['Win'] += stat['Win'] * weight
            totals['Weight'] += weight

        return totals

    def add_game(self, current):
        self.team_stats_average[current['HID']].append({
            'Date': current['Date'],
            'FieldGoalsMade': current['HFGM'],
            '3PFieldGoalsMade': current['HFG3M'],
            'FieldGoalAttempts': current['HFGA'],
            'Turnovers': current['HTOV'],
            'OffensiveRebounds': current['HORB'],
            'OpponentsDefensiveRebounds': current['ADRB'],
            'FreeThrowAttempts': current['HFTA'],
            'Score': current['HSC'],
            'Win': current['H']
        })
        self.team_stats_average[current['AID']].append({
            'Date': current['Date'],
            'FieldGoalsMade': current['AFGM'],
            '3PFieldGoalsMade': current['AFG3M'],
            'FieldGoalAttempts': current['AFGA'],
            'Turnovers': current['ATOV'],
            'OffensiveRebounds': current['AORB'],
            'OpponentsDefensiveRebounds': current['ADRB'],
            'FreeThrowAttempts': current['AFTA'],
            'Score': current['ASC'],
            'Win': current['A']
        })
        # Opponent
        self.opponent_stats_average[current['AID']].append({
            'Date': current['Date'],
            'FieldGoalsMade': current['HFGM'],
            '3PFieldGoalsMade': current['HFG3M'],
            'FieldGoalAttempts': current['HFGA'],
            'Turnovers': current['HTOV'],
            'OffensiveRebounds': current['HORB'],
            'OpponentsDefensiveRebounds': current['ADRB'],
            'FreeThrowAttempts': current['HFTA'],
            'Score': current['HSC'],
            'Win': current['H']
        })
        self.opponent_stats_average[current['HID']].append({
            'Date': current['Date'],
            'FieldGoalsMade': current['AFGM'],
            '3PFieldGoalsMade': current['AFG3M'],
            'FieldGoalAttempts': current['AFGA'],
            'Turnovers': current['ATOV'],
            'OffensiveRebounds': current['AORB'],
            'OpponentsDefensiveRebounds': current['ADRB'],
            'FreeThrowAttempts': current['AFTA'],
            'Score': current['ASC'],
            'Win': current['A']
        })

        self.cache = {}

    def get_input_data(self, date, team_id):
        cache_key = (date, team_id)
        if cache_key in self.cache:
            return self.cache[cache_key]

        if len(self.team_stats_average[team_id]) <= 5:
            return None

        stats = self._get_stats(date, self.team_stats_average[team_id])
        opp_stats = self._get_stats(date, self.opponent_stats_average[team_id])

        result = [
            (stats['FieldGoalsMade'] + 0.5 * stats['3PFieldGoalsMade']) / stats['FieldGoalAttempts'],
            stats['Turnovers'] / (stats['FieldGoalAttempts'] + 0.44 * stats['FreeThrowAttempts'] + stats['Turnovers']),
            stats['OffensiveRebounds'] / (stats['OffensiveRebounds'] + stats['OpponentsDefensiveRebounds']),
            stats['FreeThrowAttempts'] / stats['FieldGoalAttempts'],
            stats['Score'] / stats['Weight'],
            (opp_stats['FieldGoalsMade'] + 0.5 * opp_stats['3PFieldGoalsMade']) / opp_stats['FieldGoalAttempts'],
            opp_stats['Turnovers'] / (opp_stats['FieldGoalAttempts'] + 0.44 * opp_stats['FreeThrowAttempts'] + opp_stats['Turnovers']),
            opp_stats['OffensiveRebounds'] / (opp_stats['OffensiveRebounds'] + opp_stats['OpponentsDefensiveRebounds']),
            opp_stats['FreeThrowAttempts'] / opp_stats['FieldGoalAttempts'],
            opp_stats['Score'] / opp_stats['Weight']
        ]

        self.cache[cache_key] = result

        return result

total_games = len(games_df)

ff_model = FourFactor()

for index, current in games_df.iterrows():
    season = current['Season']
    home_id = current['HID']
    away_id = current['AID']
    home_score = current['HSC']
    away_score = current['ASC']

    # Make prediction
    if season in team_rosters and home_id in team_rosters[season] and away_id in team_rosters[season] and len(team_rosters[season][home_id]) >= 5 and len(team_rosters[season][away_id]) >= 5:
        home_rosters = team_rosters[season][home_id][-5:]
        away_rosters = team_rosters[season][away_id][-5:]

        home_roster = defaultdict(int)
        away_roster = defaultdict(int)

        for roster in home_rosters:
            for pid, mins in roster:
                home_roster[pid] += mins

        for roster in away_rosters:
            for pid, mins in roster:
                away_roster[pid] += mins

        home_roster = sorted(home_roster.items(), key=lambda x: x[1], reverse=True)[:15]
        away_roster = sorted(away_roster.items(), key=lambda x: x[1], reverse=True)[:15]

        while len(home_roster) < 15:
            home_roster.append([-1, 0])
        while len(away_roster) < 15:
            away_roster.append([-1, 0])

        home_total_mins = sum(x[1] for x in home_roster)
        away_total_mins = sum(x[1] for x in away_roster)

        # home_roster and away_roster both are of length 15, contain the players who play the most

        if home_total_mins >= 500 and away_total_mins >= 500:
            c_home_inputs = []
            c_home_playtimes = []
            c_away_inputs = []
            c_away_playtimes = []

            for pid, mins in home_roster:
                c_player_data = []

                if pid != -1 and pid in player_data:
                    c_player_data = copy.deepcopy(player_data[pid][-30:])
                    c_player_data.reverse()

                while len(c_player_data) < 30:
                    c_player_data.append([0] * 17)

                for i in range(len(c_player_data)):
                    c_player_data[i][0] = round(c_player_data[i][0] * (0.96 ** i), 2)

                    if c_player_data[i][0] > 0:
                        vals = ff_model.get_input_data(current['Date'], away_id)
                        if vals is not None:
                            c_player_data[i].append(1)
                            c_player_data[i].extend(vals)
                        else:
                            c_player_data[i].extend([0] * 11)
                    else:
                        c_player_data[i].extend([0] * 11)

                c_home_inputs.append(c_player_data)
                c_home_playtimes.append(mins / home_total_mins)

            for pid, mins in away_roster:
                c_player_data = []

                if pid != -1 and pid in player_data:
                    c_player_data = copy.deepcopy(player_data[pid][-30:])
                    c_player_data.reverse()

                while len(c_player_data) < 30:
                    c_player_data.append([0] * 17)

                for i in range(len(c_player_data)):
                    c_player_data[i][0] = round(c_player_data[i][0] * (0.96 ** i), 2)

                    if c_player_data[i][0] > 0:
                        vals = ff_model.get_input_data(current['Date'], home_id)
                        if vals is not None:
                            c_player_data[i].append(1)
                            c_player_data[i].extend(vals)
                        else:
                            c_player_data[i].extend([0] * 11)
                    else:
                        c_player_data[i].extend([0] * 11)

                c_away_inputs.append(c_player_data)
                c_away_playtimes.append(mins / away_total_mins)

            keys.append(index)
            home_inputs.append(c_home_inputs)
            home_playtimes.append(c_home_playtimes)
            away_inputs.append(c_away_inputs)
            away_playtimes.append(c_away_playtimes)
            outputs.append(abs(home_score - away_score) ** 0.7 * (1 if home_score > away_score else -1))

    # Log data
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

    mapped_home_players = [{
        'pid': row['Player'],
        'mins': row['MIN'],
        'inputs': [
            1,                          # Whether player is part of home team
            row['MIN'],
            row['PTS'] / row['MIN'],    # Points
            row['ORB'] / row['MIN'],    # Offensive rebounds
            row['DRB'] / row['MIN'],    # Defensive rebounds
            row['AST'] / row['MIN'],    # Assists
            row['STL'] / row['MIN'],    # Steals
            row['BLK'] / row['MIN'],    # Blocks
            row['FGA'] / row['MIN'],    # Field goal attempts
            row['FTA'] / row['MIN'],    # Free throw attempts
            row['TOV'] / row['MIN'],    # Turnovers
            row['PF'] / row['MIN'],     # Personal fouls
            (row['FGM'] + 0.5 * row['FG3M']) / (row['FGA'] + 0.00001),
            row['TOV'] / (row['FGA'] + 0.44 * row['FTA'] + row['TOV'] + 0.00001),
            row['ORB'] / (row['ORB'] + row['DRB'] + 0.00001),
            row['FTA'] / (row['FGA'] + 0.00001)
        ]
    } for _, row in home_players.iterrows()]
    mapped_away_players = [{
        'pid': row['Player'],
        'mins': row['MIN'],
        'inputs': [
            0,                          # Whether player is part of home team
            row['MIN'],
            row['PTS'] / row['MIN'],    # Points
            row['ORB'] / row['MIN'],    # Offensive rebounds
            row['DRB'] / row['MIN'],    # Defensive rebounds
            row['AST'] / row['MIN'],    # Assists
            row['STL'] / row['MIN'],    # Steals
            row['BLK'] / row['MIN'],    # Blocks
            row['FGA'] / row['MIN'],    # Field goal attempts
            row['FTA'] / row['MIN'],    # Free throw attempts
            row['TOV'] / row['MIN'],    # Turnovers
            row['PF'] / row['MIN'],     # Personal fouls
            (row['FGM'] + 0.5 * row['FG3M']) / (row['FGA'] + 0.00001),
            row['TOV'] / (row['FGA'] + 0.44 * row['FTA'] + row['TOV'] + 0.00001),
            row['ORB'] / (row['ORB'] + row['DRB'] + 0.00001),
            row['FTA'] / (row['FGA'] + 0.00001)
        ]
    } for _, row in away_players.iterrows()]

    for data in [*mapped_home_players, *mapped_away_players]:
        if not any(math.isnan(x) for x in data['inputs']):
            player_data[data['pid']].append([data['mins'], *data['inputs']])

    ff_model.add_game(current)

    if (index + 1) % (total_games // 1000) == 0:
        progress = (index + 1) / total_games * 100
        print(f'Progress: {progress:.1f}%')

with open('temp/keys.json', 'w') as json_file:
    json.dump(keys, json_file)

np.save('temp/nn_home_inputs.npy', np.array(home_inputs).astype(np.float32))
np.save('temp/nn_away_inputs.npy', np.array(away_inputs).astype(np.float32))
np.save('temp/nn_home_playtimes.npy', np.array(home_playtimes).astype(np.float32))
np.save('temp/nn_away_playtimes.npy', np.array(away_playtimes).astype(np.float32))
np.save('temp/nn_outputs.npy', np.array(outputs))

print('Done')
