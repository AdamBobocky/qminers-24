import math
from collections import defaultdict

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

class Elo:
    def __init__(self, k_factor=0.15, home_factor=0.5):
        self.k_factor = k_factor
        self.home_factor = home_factor

        self.elo_map = defaultdict(float)

    def pre_add_game(self, current, current_players):
        pass

    def add_game(self, current, current_players):
        home_id = current['HID']
        away_id = current['AID']

        elo_prediction = sigmoid(self.get_input_data(home_id, away_id, 0, 0)[0])

        self.elo_map[home_id] += self.k_factor * (current['H'] - elo_prediction)
        self.elo_map[away_id] += self.k_factor * (current['A'] - (1 - elo_prediction))

    def get_input_data(self, home_id, away_id, season, date):
        return [
            self.elo_map[home_id] - self.elo_map[away_id] + self.home_factor
        ]

