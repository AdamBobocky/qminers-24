import pandas as pd

import sys

sys.path.append('.')

from trueskill_model import Model
from environment_simpler import Environment

games = pd.read_csv('./data/games.csv', index_col=0)
games['Date'] = pd.to_datetime(games['Date'])
games['Open'] = pd.to_datetime(games['Open'])

players = pd.read_csv('./data/players.csv', index_col=0)
players['Date'] = pd.to_datetime(players['Date'])

env = Environment(games, players, Model())

evaluation = env.run()
