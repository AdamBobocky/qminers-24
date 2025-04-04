import json
import math
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from pythagorean.model import Pythagorean
from four_factor.model import FourFactor
from nate_silver_elo.model import NateSilverElo
from gradient_descent.model import GradientDescent
from exhaustion.model import Exhaustion
from neural_network.model import NeuralNetwork

def inverse_sigmoid(x):
    return math.log(x / (1 - x))

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

class Model:
    def __init__(self, debug_mode=False):
        self.debug_mode = debug_mode

        # Hyperparameters
        self.ensamble_required_n = 2000
        nate_silver_elo = NateSilverElo()
        self.model_list = [
            Pythagorean(),      # 0.8464013687751296, -0.06697869116809001
            FourFactor(),       # -0.037466710615323806
            GradientDescent(),  # 0.8539410540350695
            Exhaustion(),       # -0.30556362733411674
            nate_silver_elo,    # 0.002608191859624124
            NeuralNetwork(nate_silver_elo)
        ]
        # End

        self.coef_list = []

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

        for model_i, model in enumerate(self.model_list):
            data = model.get_input_data(home_id, away_id, season, date)

            if data is None:
                print('\nmodel_i:', model_i)

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

    def _game_increment(self, idx, current, current_players):
        season = current['Season']
        date = current['Date']
        home_id = current['HID']
        away_id = current['AID']
        home_win = current['H']
        score_diff = current['HSC'] - current['ASC']
        year = int(str(current['Date'])[0:4])

        if year >= 2002:
            input_arr = self._get_input_features(home_id, away_id, season, date)

            if input_arr is not None:
                self.past_pred.append([*input_arr, score_diff])
                self.ensamble_retrain -= 1

        self._handle_metrics(idx, current)

        if year >= 2000:
            # Let the models create training frames before new data arrives
            for model in self.model_list:
                model.pre_add_game(current, current_players)

            # Add new data to the models
            for model in self.model_list:
                model.add_game(current, current_players)

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
        bets = pd.DataFrame(data=np.zeros((len(opps), 2)), columns=['BetH', 'BetA'], index=opps.index)

        summ_date = summary.iloc[0]['Date']
        min_bet = summary.iloc[0]['Min_bet']
        max_bet = summary.iloc[0]['Max_bet']
        bankroll = summary.iloc[0]['Bankroll']

        try:
            games_increment, players_increment = inc

            if self.debug_mode:
                self._print_metrics()

                with open('src/meta_model/data.json', 'w') as json_file:
                    json.dump(self.pred_list, json_file, indent=2)

            done = 0
            total = len(games_increment)

            for idx in games_increment.index:
                current = games_increment.loc[idx]
                current_players = players_increment[(players_increment['Game'] == idx) & (players_increment['MIN'] >= 3)]

                self._game_increment(idx, current, current_players)
                done += 1
                if done % 100 == 0:
                    print(f'{done} / {total}')

            for i in opps.index:
                current = opps.loc[i]

                season = current['Season']
                date = current['Date']
                home_id = current['HID']
                away_id = current['AID']
                playoff = current['POFF'] == 1
                days_until = (date - summ_date).days

                if len(self.past_pred) >= self.ensamble_required_n:
                    input_arr = self._get_input_features(home_id, away_id, season, date)

                    if input_arr is not None:
                        if self.ensamble_retrain <= 0:
                            self.ensamble_retrain = 400
                            np_array = np.array(self.past_pred)
                            sample_weights = np.exp(-0.0003 * np.arange(len(self.past_pred)))
                            self.ensamble = LinearRegression()
                            self.ensamble.fit(np_array[:, :-1], np_array[:, -1], sample_weight=sample_weights[::-1])

                            self.coef_list.append({
                                'index': i,
                                'date': str(date),
                                'coefs': self.ensamble.coef_.tolist(),
                                'intercept': self.ensamble.intercept_.tolist(),
                                'sum_weight': sample_weights.sum(),
                                'len': len(self.past_pred)
                            })

                            with open('src/meta_model/coef_list.json', 'w') as json_file:
                                json.dump(self.coef_list, json_file, indent=2)

                        self.bet_metrics['opps'] += 1

                        pred = self.ensamble.predict(np.array([input_arr]))[0]

                        self.prediction_map[i] = pred
                        self.input_map[i] = input_arr
                        self.coef_map[i] = [self.ensamble.intercept_.tolist(), *self.ensamble.coef_.tolist()]

        except Exception:
            pass

        return bets
