import math
import json
import pandas as pd
import numpy as np
import copy
from collections import defaultdict
from datetime import datetime

games_df = pd.read_csv('data/games.csv')

keys = []
team_data = defaultdict(list)

home_inputs = []
away_inputs = []
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

for index, current in games_df.iterrows():
    season = current['Season']
    home_id = current['HID']
    away_id = current['AID']
    home_score = current['HSC']
    away_score = current['ASC']
    date = datetime.strptime(current['Date'], '%Y-%m-%d')

    # Make prediction
    if len(team_data[home_id]) > 0 and len(team_data[away_id]) > 0:
        home_instance = copy.deepcopy(team_data[home_id][-60:])

        for i in range(len(home_instance)):
            home_instance[i][0] = 0.9965 ** abs((date - home_instance[i][0]).days)

        while len(home_instance) < 60:
            home_instance.append([0] * 38)

        away_instance = copy.deepcopy(team_data[away_id][-60:])

        for i in range(len(away_instance)):
            away_instance[i][0] = 0.9965 ** abs((date - away_instance[i][0]).days)

        while len(away_instance) < 60:
            away_instance.append([0] * 38)

        keys.append(index)
        home_inputs.append(home_instance)
        away_inputs.append(away_instance)
        outputs.append((abs(home_score - away_score) + 3) ** 0.7 * (1 if home_score > away_score else -1))

    # Log data
    team_data[home_id].append([
        date,
        elo.get_team_strength(home_id, True, season) / 100,
        elo.get_team_strength(away_id, False, season) / 100,
        1,
        home_score,
        away_score,
        current['POFF'],
        current['Season'],
        current['OddsH'],
        current['OddsA'],
        current['HFGM'],
        current['AFGM'],
        current['HFGA'],
        current['AFGA'],
        current['HFG3M'],
        current['AFG3M'],
        current['HFG3A'],
        current['AFG3A'],
        current['HFTM'],
        current['AFTM'],
        current['HFTA'],
        current['AFTA'],
        current['HORB'],
        current['AORB'],
        current['HDRB'],
        current['ADRB'],
        current['HRB'],
        current['ARB'],
        current['HAST'],
        current['AAST'],
        current['HSTL'],
        current['ASTL'],
        current['HBLK'],
        current['ABLK'],
        current['HTOV'],
        current['ATOV'],
        current['HPF'],
        current['APF']
    ])
    team_data[away_id].append([
        date,
        elo.get_team_strength(away_id, False, season) / 100,
        elo.get_team_strength(home_id, True, season) / 100,
        0,
        away_score,
        home_score,
        current['POFF'],
        current['Season'],
        current['OddsH'],
        current['OddsA'],
        current['HFGM'],
        current['AFGM'],
        current['HFGA'],
        current['AFGA'],
        current['HFG3M'],
        current['AFG3M'],
        current['HFG3A'],
        current['AFG3A'],
        current['HFTM'],
        current['AFTM'],
        current['HFTA'],
        current['AFTA'],
        current['HORB'],
        current['AORB'],
        current['HDRB'],
        current['ADRB'],
        current['HRB'],
        current['ARB'],
        current['HAST'],
        current['AAST'],
        current['HSTL'],
        current['ASTL'],
        current['HBLK'],
        current['ABLK'],
        current['HTOV'],
        current['ATOV'],
        current['HPF'],
        current['APF']
    ])

    elo.add_game(current)

    if (index + 1) % (total_games // 1000) == 0:
        progress = (index + 1) / total_games * 100
        print(f'Progress: {progress:.1f}%')

with open('temp/team_keys.json', 'w') as json_file:
    json.dump(keys, json_file)

np.save('temp/team_nn_home_inputs.npy', np.array(home_inputs).astype(np.float32))
np.save('temp/team_nn_away_inputs.npy', np.array(away_inputs).astype(np.float32))
np.save('temp/team_nn_outputs.npy', np.array(outputs))

print('Done')
