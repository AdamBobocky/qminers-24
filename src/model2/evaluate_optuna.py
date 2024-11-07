import optuna
import pandas as pd
import sys
import json

sys.path.append(".")

from model import Model
from environment import Environment

games = pd.read_csv("./data/games.csv", index_col=0)
games["Date"] = pd.to_datetime(games["Date"])
games["Open"] = pd.to_datetime(games["Open"])

players = pd.read_csv("./data/players.csv", index_col=0)
players["Date"] = pd.to_datetime(players["Date"])

def objective(trial):
    prior_sigma = trial.suggest_float('prior_sigma', 0.5, 128.0)
    monthly_decay = trial.suggest_float('monthly_decay', 0.6, 1.0)
    season_reset_mult = trial.suggest_float('season_reset_mult', 0.05, 4.0)

    env = Environment(games, players, Model(prior_sigma, monthly_decay, season_reset_mult), init_bankroll=1000, min_bet=5, max_bet=100)
    # env = Environment(games, players, Model(prior_sigma, monthly_decay), init_bankroll=1000, min_bet=5, max_bet=100, end_date=pd.Timestamp('1985-01-01'))

    evaluation = env.run()

    history = env.get_history()

    with open('mse.json', 'r') as file:
        mse = json.load(file)['mse']

    print('\n\nmse:', mse)

    return mse

study = optuna.create_study(
    direction='minimize',
    storage='sqlite:///my_study.db'
)
study.optimize(objective, n_trials=100)

best_params = study.best_params
best_score = study.best_value
