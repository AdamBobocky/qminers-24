import json
import math
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.linear_model import LogisticRegression

K_FACTOR = 0.15
HOME_FACTOR = 0.5

with open('temp/xgboost_prediction_map.json', 'r') as json_file:
    xgboost_prediction_map = json.load(json_file)

class Model:
    elo_map = {}
    games_data = []
    predictions = []
    metrics = {
        'ensamble_mse': 0,
        'lr_mse': 0,
        'elo_mse': 0,
        'odds_mse': 0,
        'n': 0
    }

    def place_bets(self, summary: pd.DataFrame, opps: pd.DataFrame, inc: tuple[pd.DataFrame, pd.DataFrame]):
        games_increment, players_increment = inc
        print(summary['Date'])
        # print(players_increment)
        # print(opps) # Contains betting opportunities
        # print(games_increment) # New games data
        # print(players_increment) # New players data

        self.games_data.append(games_increment)

        for i in games_increment.index:
            current = games_increment.loc[i]

            home_id = current['HID']
            away_id = current['AID']
            home_win = current['H']
            neutral = current['N']

            for current_id in [home_id, away_id]:
                if current_id not in self.elo_map:
                    self.elo_map[current_id] = 0

            prediction = sigmoid(self.elo_map[home_id] - self.elo_map[away_id] + (1 - neutral) * HOME_FACTOR)

            self.elo_map[home_id] += K_FACTOR * (home_win - prediction)
            self.elo_map[away_id] += K_FACTOR * ((1 - home_win) - (1 - prediction))

        for i in opps.index:
            current = opps.loc[i]

            home_id = current['HID']
            away_id = current['AID']
            home_odds = current['OddsH']
            away_odds = current['OddsA']
            neutral = current['N']

            for current_id in [home_id, away_id]:
                if current_id not in self.elo_map:
                    self.elo_map[current_id] = 0

            prediction = sigmoid(self.elo_map[home_id] - self.elo_map[away_id] + (1 - neutral) * HOME_FACTOR)
            odds_prediction = 1 / home_odds / (1 / home_odds + 1 / away_odds)

            self.predictions.append({
                'index': i,
                'elo_pred': prediction,
                'odds_pred': odds_prediction,
                'home_odds': home_odds,
                'away_odds': away_odds
            })

    def end(self):
        merge = pd.concat(self.games_data)

        backtests = {
            'pnl_0%': 0,
            'bets_0%': 0,
            'vig_0%': 0,
            'odds_0%': 0,
            'pnl_10%': 0,
            'bets_10%': 0,
            'vig_10%': 0,
            'odds_10%': 0,
            'pnl_20%': 0,
            'bets_20%': 0,
            'vig_20%': 0,
            'odds_20%': 0
        }

        factorA = []
        factorB = []
        target = []

        for current in self.predictions:
            if str(current['index']) in xgboost_prediction_map and current['index'] in merge.index:
                xg_pred = xgboost_prediction_map[str(current['index'])]

                home_win = merge.at[current['index'], 'H']
                away_win = merge.at[current['index'], 'A']
                elo_pred = current['elo_pred']
                odds_pred = current['odds_pred']
                home_odds = current['home_odds']
                away_odds = current['away_odds']

                ensamble_pred = sigmoid(0.09445932 + inverse_sigmoid(xg_pred) * 0.75088559 + inverse_sigmoid(elo_pred) * 0.88550342)

                factorA.append(inverse_sigmoid(xg_pred))
                factorB.append(inverse_sigmoid(elo_pred))
                target.append(home_win)

                self.metrics['ensamble_mse'] += (ensamble_pred - home_win) ** 2
                self.metrics['elo_mse'] += (elo_pred - home_win) ** 2
                self.metrics['odds_mse'] += (odds_pred - home_win) ** 2
                self.metrics['n'] += 1

                if ensamble_pred * home_odds > 1:
                    backtests['pnl_0%'] -= 1
                    backtests['odds_0%'] += home_odds
                    if home_win:
                        backtests['pnl_0%'] += home_odds
                    backtests['vig_0%'] += 1 / home_odds + 1 / away_odds - 1
                    backtests['bets_0%'] += 1
                if (1 - ensamble_pred) * away_odds > 1:
                    backtests['pnl_0%'] -= 1
                    backtests['odds_0%'] += away_odds
                    if away_win:
                        backtests['pnl_0%'] += away_odds
                    backtests['vig_0%'] += 1 / home_odds + 1 / away_odds - 1
                    backtests['bets_0%'] += 1
                #
                if ensamble_pred * home_odds > 1.1:
                    backtests['pnl_10%'] -= 1
                    backtests['odds_10%'] += home_odds
                    if home_win:
                        backtests['pnl_10%'] += home_odds
                    backtests['vig_10%'] += 1 / home_odds + 1 / away_odds - 1
                    backtests['bets_10%'] += 1
                if (1 - ensamble_pred) * away_odds > 1.1:
                    backtests['pnl_10%'] -= 1
                    backtests['odds_10%'] += away_odds
                    if away_win:
                        backtests['pnl_10%'] += away_odds
                    backtests['vig_10%'] += 1 / home_odds + 1 / away_odds - 1
                    backtests['bets_10%'] += 1
                #
                if ensamble_pred * home_odds > 1.2:
                    backtests['pnl_20%'] -= 1
                    backtests['odds_20%'] += home_odds
                    if home_win:
                        backtests['pnl_20%'] += home_odds
                    backtests['vig_20%'] += 1 / home_odds + 1 / away_odds - 1
                    backtests['bets_20%'] += 1
                if (1 - ensamble_pred) * away_odds > 1.2:
                    backtests['pnl_20%'] -= 1
                    backtests['odds_20%'] += away_odds
                    if away_win:
                        backtests['pnl_20%'] += away_odds
                    backtests['vig_20%'] += 1 / home_odds + 1 / away_odds - 1
                    backtests['bets_20%'] += 1

        X = np.column_stack((factorA, factorB))
        y = np.array(target)

        model = LogisticRegression()
        model.fit(X, y)
        y_pred = model.predict_proba(X)[:, 1]

        print(model)
        print('Model Parameters:')
        print(model.get_params())
        print('Model Coefficients:')
        print(model.coef_)
        print('Model Intercept:')
        print(model.intercept_)

        for j in range(len(target)):
            self.metrics['lr_mse'] += (y_pred[j] - target[j]) ** 2

        print('lr_mse:', self.metrics['lr_mse'] / self.metrics['n'])
        print('ensamble_mse:', self.metrics['ensamble_mse'] / self.metrics['n'])
        print('elo_mse:', self.metrics['elo_mse'] / self.metrics['n'])
        print('odds_mse:', self.metrics['odds_mse'] / self.metrics['n'])
        print(backtests)

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def inverse_sigmoid(x):
    return np.log(x / (1 - x))
