import math
import json
import pandas as pd
import numpy as np
import copy
from collections import defaultdict
from datetime import datetime

players_df = pd.read_csv('data-merge-round2/players.csv')
games_df = pd.read_csv('data-merge-round2/games.csv')

keys = []
player_data = defaultdict(list)
player_teams = defaultdict(int)
team_rosters = {}

home_inputs = []
away_inputs = []
home_playtimes = []
away_playtimes = []
outputs = []

class NateSilverElo:
    def __init__(self):
        self.elo_map = defaultdict(float)
        self.last_season = -1

    def _new_season(self):
        for key in self.elo_map:
            self.elo_map[key] *= 0.75

    def _win_probability(self, x):
        return 1 / (1 + (math.exp(-x / 175)))

    def add_game(self, current):
        season = current['Season']
        home_id = current['HID']
        away_id = current['AID']
        home_score = current['HSC']
        away_score = current['ASC']

        if season > self.last_season:
            self.last_season = season
            self._new_season()

        home_prediction = self._win_probability(self.elo_map[home_id] + 100 - self.elo_map[away_id])
        away_prediction = 1 - home_prediction

        k_factor = self.get_k_factor(home_score - away_score, self.elo_map[home_id] + 100, self.elo_map[away_id])

        self.elo_map[home_id] += k_factor * (current['H'] - home_prediction)
        self.elo_map[away_id] += k_factor * (current['A'] - away_prediction)

    def get_team_strength(self, team_id, is_home, season):
        if season > self.last_season:
            self.last_season = season
            self._new_season()

        return self.elo_map[team_id] + 100 * (0.5 if is_home else -0.5)

    def get_k_factor(self, score_difference, elo_home, elo_away):
        if score_difference > 0:
            return 20 * (score_difference + 3) ** 0.8 / (7.5 + 0.006 * (elo_home - elo_away))
        else:
            return 20 * (-score_difference + 3) ** 0.8 / (7.5 + 0.006 * (elo_away - elo_home))

total_games = len(games_df)

elo = NateSilverElo()

INPUTS_DIM = 19

def row_to_inputs(row, am_home, my_id, opponent_id, season):
    return [
        elo.get_team_strength(my_id, am_home, season) / 100,
        elo.get_team_strength(opponent_id, not am_home, season) / 100,
        1 if am_home else 0,        # Whether player is part of home team
        row['MIN'],
        row['FGM'],

        row['FGA'],
        row['FG3M'],
        row['FG3A'],
        row['FTM'],
        row['FTA'],

        row['ORB'],
        row['DRB'],
        row['RB'],
        row['AST'],
        row['STL'],

        row['BLK'],
        row['TOV'],
        row['PF'],
        row['PTS']
    ]

for index, current in games_df.iterrows():
    season = current['Season']
    home_id = current['HID']
    away_id = current['AID']
    home_score = current['HSC']
    away_score = current['ASC']
    date = datetime.strptime(current['Date'], '%Y-%m-%d')

    if index > 17500:
    # if index > 5000:
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

            home_roster = sorted(home_roster.items(), key=lambda x: x[1], reverse=True)[:12]
            away_roster = sorted(away_roster.items(), key=lambda x: x[1], reverse=True)[:12]

            while len(home_roster) < 12:
                home_roster.append([-1, 0])
            while len(away_roster) < 12:
                away_roster.append([-1, 0])

            home_total_mins = sum(x[1] for x in home_roster)
            away_total_mins = sum(x[1] for x in away_roster)

            # home_roster and away_roster both are of length 12, contain the players who play the most

            if home_total_mins >= 500 and away_total_mins >= 500:
                c_home_inputs = []
                c_home_playtimes = []
                c_away_inputs = []
                c_away_playtimes = []

                for pid, mins in home_roster:
                    c_player_data = []

                    if pid != -1 and pid in player_data:
                        c_player_data = copy.deepcopy(player_data[pid][-40:])

                    for i in range(len(c_player_data)):
                        point_date, point_mins = c_player_data[i][0]
                        time_weight = 0.9965 ** abs((date - point_date).days)
                        c_player_data[i][0] = round(point_mins * time_weight, 3) # Apply time decay

                    while len(c_player_data) < 40:
                        c_player_data.append([0] * (INPUTS_DIM + 2))

                    c_home_inputs.append(c_player_data)
                    c_home_playtimes.append(mins / home_total_mins)

                for pid, mins in away_roster:
                    c_player_data = []

                    if pid != -1 and pid in player_data:
                        c_player_data = copy.deepcopy(player_data[pid][-40:])

                    for i in range(len(c_player_data)):
                        point_date, point_mins = c_player_data[i][0]
                        time_weight = 0.9965 ** abs((date - point_date).days)
                        c_player_data[i][0] = round(point_mins * time_weight, 3) # Apply time decay

                    while len(c_player_data) < 40:
                        c_player_data.append([0] * (INPUTS_DIM + 2))

                    c_away_inputs.append(c_player_data)
                    c_away_playtimes.append(mins / away_total_mins)

                keys.append(index)
                home_inputs.append(c_home_inputs)
                home_playtimes.append(c_home_playtimes)
                away_inputs.append(c_away_inputs)
                away_playtimes.append(c_away_playtimes)
                outputs.append((abs(home_score - away_score) + 3) ** 0.7 * (1 if home_score > away_score else -1))

        # Log data
        game_players = players_df[(players_df['Game'] == index) & (players_df['MIN'] >= 3)]

        players_on_a_team_map = {}

        for _, player in game_players.iterrows():
            key = f"{player['Player']}|{player['Team']}"
            players_on_a_team_map[player['Player']] = math.log(1 + player_teams[key])
            player_teams[key] += 1

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
            'inputs': row_to_inputs(row, True, home_id, away_id, season)
        } for _, row in home_players.iterrows()]
        mapped_away_players = [{
            'pid': row['Player'],
            'mins': row['MIN'],
            'inputs': row_to_inputs(row, False, away_id, home_id, season)
        } for _, row in away_players.iterrows()]

        for data in [*mapped_home_players, *mapped_away_players]:
            if not any(math.isnan(x) for x in data['inputs']):
                player_data[data['pid']].append([[date, data['mins']], *data['inputs'], players_on_a_team_map[data['pid']]])

    elo.add_game(current)

    if (index + 1) % (total_games // 1000) == 0:
        progress = (index + 1) / total_games * 100
        print(f'Progress: {progress:.1f}%')

with open('temp/keys.json', 'w') as json_file:
    json.dump(keys, json_file)

np_array_home_inputs = np.array(home_inputs).astype(np.float32)
np_array_away_inputs = np.array(away_inputs).astype(np.float32)

np.save('temp/nn_home_inputs.npy', np_array_home_inputs)
np.save('temp/nn_away_inputs.npy', np_array_away_inputs)
np.save('temp/nn_home_playtimes.npy', np.array(home_playtimes).astype(np.float32))
np.save('temp/nn_away_playtimes.npy', np.array(away_playtimes).astype(np.float32))
np.save('temp/nn_outputs.npy', np.array(outputs))

print('Done')
