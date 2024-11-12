import json
import math
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from pythagorean.model import Pythagorean
from four_factor.model import FourFactor
from elo.model import Elo
from gradient_descent.model import GradientDescent

class Model:
    def __init__(self, debug_mode=False):
        self.debug_mode = debug_mode

        # Hyperparameters
        self.ensamble_required_n = 2000
        self.model_list = [
            Pythagorean(),
            FourFactor(),
            Elo(),
            GradientDescent()
        ]
        # End

        self.prediction_map = {}
        self.input_map = {}
        self.coef_map = {}
        self.past_pred = []
        self.ensamble = None
        self.ensamble_retrain = 0

        self.pred_list = []

        self.pred_metrics = {
            'my_ba': 0,
            'mkt_ba': 0,
            'my_mse': 0,
            'mkt_mse': 0,
            'corr_me': [],
            'corr_mkt': [],
            'n': 0
        }

        self.bet_metrics = {
            'exp_pnl': 0,
            'opps': 0,
            'count': 0,
            'volume': 0,
            'sum_odds': 0
        }

    def _get_input_features(self, home_id, away_id, season, date):
        input_data = []

        for model in self.model_list:
            data = model.get_input_data(home_id, away_id, season, date)

            if data is None:
                return None

            input_data = [
                *input_data,
                *data
            ]

        return input_data

    def _handle_metrics(self, idx, current):
        if idx in self.prediction_map:
            pred = self.prediction_map[idx]

            home_win = current['H']
            odds_home = current['OddsH']
            odds_away = current['OddsA']
            overround = 1 / odds_home + 1 / odds_away
            mkt_pred = 1 / odds_home / overround

            if pred == 0.5:
                self.pred_metrics['my_ba'] += 0.5
            elif (pred > 0.5) == home_win:
                self.pred_metrics['my_ba'] += 1

            if mkt_pred == 0.5:
                self.pred_metrics['mkt_ba'] += 0.5
            elif (mkt_pred > 0.5) == home_win:
                self.pred_metrics['mkt_ba'] += 1

            self.pred_metrics['my_mse'] += (pred - home_win) ** 2
            self.pred_metrics['mkt_mse'] += (mkt_pred - home_win) ** 2
            self.pred_metrics['n'] += 1

            self.pred_metrics['corr_me'].append(pred)
            self.pred_metrics['corr_mkt'].append(mkt_pred)

            self.pred_list.append({
                'index': str(idx),
                'neutral': int(current['N']),
                'playoff': int(current['POFF']),
                'date': str(current['Date']),
                'season': int(current['Season']),
                'score': int(current['HSC'] - current['ASC']),
                'my_pred': pred,
                'mkt_pred': mkt_pred,
                'odds_home': float(odds_home),
                'odds_away': float(odds_away),
                'outcome': int(home_win),
                'inputs': self.input_map[idx],
                'coefs': self.coef_map[idx]
            })

    def _game_increment(self, idx, current):
        season = current['Season']
        date = current['Date']
        home_id = current['HID']
        away_id = current['AID']
        home_win = current['H']

        if int(str(current['Date'])[0:4]) >= 1990:
            input_arr = self._get_input_features(home_id, away_id, season, date)

            if input_arr is not None:
                self.past_pred.append([*input_arr, home_win])
                self.ensamble_retrain -= 1

        self._handle_metrics(idx, current)

        for model in self.model_list:
            model.add_game(current)

    def _print_metrics(self):
        print('')

        if self.bet_metrics['count'] > 0:
            print('Opps:', self.bet_metrics['opps'], 'Bets:', self.bet_metrics['count'], 'Volume:', self.bet_metrics['volume'], 'Avg odds:', self.bet_metrics['sum_odds'] / self.bet_metrics['count'], 'Exp avg P&L:', self.bet_metrics['exp_pnl'] / self.bet_metrics['count'])

        if self.pred_metrics['n'] > 0:
            r = np.corrcoef(self.pred_metrics['corr_me'], self.pred_metrics['corr_mkt'])[0, 1]
            r_squared = r ** 2

            print('my_ba    ', self.pred_metrics['my_ba'] / self.pred_metrics['n'], self.pred_metrics['n'])
            print('mkt_ba   ', self.pred_metrics['mkt_ba'] / self.pred_metrics['n'], self.pred_metrics['n'])
            print('my_mse   ', self.pred_metrics['my_mse'] / self.pred_metrics['n'], self.pred_metrics['n'])
            print('mkt_mse  ', self.pred_metrics['mkt_mse'] / self.pred_metrics['n'], self.pred_metrics['n'])
            print('ba corr  ', np.sum(np.round(self.pred_metrics['corr_me']) == np.round(self.pred_metrics['corr_mkt'])) / len(self.pred_metrics['corr_mkt']))
            print('corr r   ', r)
            print('corr r2  ', r_squared)

    def place_bets(self, summary: pd.DataFrame, opps: pd.DataFrame, inc: tuple[pd.DataFrame, pd.DataFrame]):
        games_increment, players_increment = inc

        if self.debug_mode:
            self._print_metrics()

            with open('src/meta_model/data.json', 'w') as json_file:
                json.dump(self.pred_list, json_file, indent=2)

        for idx in games_increment.index:
            current = games_increment.loc[idx]

            self._game_increment(idx, current)

        min_bet = summary.iloc[0]['Min_bet']
        max_bet = summary.iloc[0]['Max_bet']

        bets = pd.DataFrame(data=np.zeros((len(opps), 2)), columns=['BetH', 'BetA'], index=opps.index)

        for i in opps.index:
            current = opps.loc[i]

            season = current['Season']
            date = current['Date']
            home_id = current['HID']
            away_id = current['AID']

            input_arr = self._get_input_features(home_id, away_id, season, date)

            if input_arr is not None and len(self.past_pred) >= self.ensamble_required_n:
                if self.ensamble_retrain <= 0:
                    self.ensamble_retrain = 200
                    np_array = np.array(self.past_pred)
                    sample_weights = np.exp(-0.0003 * np.arange(len(self.past_pred)))
                    self.ensamble = LogisticRegression(max_iter=10000)
                    self.ensamble.fit(np_array[:, :-1], np_array[:, -1], sample_weight=sample_weights[::-1])

                self.bet_metrics['opps'] += 1

                pred = self.ensamble.predict_proba(np.array([input_arr]))[0, 1]

                self.prediction_map[i] = pred
                self.input_map[i] = input_arr
                self.coef_map[i] = [self.ensamble.intercept_.tolist(), *self.ensamble.coef_.tolist()]

                odds_home = current['OddsH']
                odds_away = current['OddsA']

                min_home_odds = (1 / pred - 1) * 1.1 + 1 + 0.03
                min_away_odds = (1 / (1 - pred) - 1) * 1.1 + 1 + 0.03

                if odds_home >= min_home_odds:
                    bets.at[i, 'BetH'] = min_bet

                    self.bet_metrics['exp_pnl'] += pred * odds_home - 1
                    self.bet_metrics['volume'] += min_bet
                    self.bet_metrics['count'] += 1
                    self.bet_metrics['sum_odds'] += odds_home

                if odds_away >= min_away_odds:
                    bets.at[i, 'BetA'] = min_bet

                    self.bet_metrics['exp_pnl'] += (1 - pred) * odds_away - 1
                    self.bet_metrics['volume'] += min_bet
                    self.bet_metrics['count'] += 1
                    self.bet_metrics['sum_odds'] += odds_away

        return bets
