import json
import optuna

with open('src/meta_model/data.json', 'r') as file:
    data = json.load(file)

def objective(trial):
    home_mult = trial.suggest_float('home_mult', 1.0, 2.0)
    home_flat = trial.suggest_float('home_flat', 0, 2.0)
    away_mult = trial.suggest_float('away_mult', 1.0, 2.0)
    away_flat = trial.suggest_float('away_flat', 0, 2.0)

    pnl = 0

    for el in data:
        pred = el['my_pred']
        odds_home = el['odds_home']
        odds_away = el['odds_away']
        outcome = el['outcome']

        min_home_odds = (1 / pred - 1) * home_mult + 1 + home_flat
        min_away_odds = (1 / (1 - pred) - 1) * away_mult + 1 + away_flat

        if odds_home > min_home_odds:
            pnl += (outcome * odds_home) - 1

        if odds_away > min_away_odds:
            pnl += ((1 - outcome) * odds_away) - 1

    return pnl

study = optuna.create_study(
    direction='maximize',
    storage='sqlite:///my_study.db'
)
study.optimize(objective, n_trials=4000)
