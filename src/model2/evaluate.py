import pandas as pd

import sys

sys.path.append(".")

from model import Model
from environment import Environment

games = pd.read_csv("./data/games.csv", index_col=0)
games["Date"] = pd.to_datetime(games["Date"])
games["Open"] = pd.to_datetime(games["Open"])

players = pd.read_csv("./data/players.csv", index_col=0)
players["Date"] = pd.to_datetime(players["Date"])

env = Environment(games, players, Model(monthly_decay=0.75, season_reset_mult=0.8), init_bankroll=1000, min_bet=5, max_bet=100)

evaluation = env.run()

print()
print(f'Final bankroll: {env.bankroll:.2f}')

history = env.get_history()

print(history)
