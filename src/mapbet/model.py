import json
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.linear_model import LogisticRegression

with open('temp/predictions.json', 'r') as f:
    predictions = json.load(f)

class Model:
    def __init__(self):
        self.prediction_map = {}
        self.past_pred = []
        self.corr_me = []
        self.corr_mkt = []

        self.metrics = {
            'my_ba': 0,
            'mkt_ba': 0,
            'my_mse': 0,
            'mkt_mse': 0,
            'n': 0
        }

        self.bet_opps = 0
        self.exp_pnl = 0
        self.bet_volume = 0
        self.bet_count = 0
        self.bet_sum_odds = 0

    def print_metrics(self):
        if self.bet_count > 0:
            print('\nOpps:', self.bet_opps, 'Bets:', self.bet_count, 'Volume:', self.bet_volume, 'Avg odds:', self.bet_sum_odds / self.bet_count, 'Exp avg P&L:', self.exp_pnl / self.bet_count)

        if self.metrics['n'] > 0:
            r = np.corrcoef(self.corr_me, self.corr_mkt)[0, 1]
            r_squared = r ** 2

            print('\nmy_ba    ', self.metrics['my_ba'] / self.metrics['n'], self.metrics['n'])
            print('mkt_ba   ', self.metrics['mkt_ba'] / self.metrics['n'], self.metrics['n'])
            print('my_mse   ', self.metrics['my_mse'] / self.metrics['n'], self.metrics['n'])
            print('mkt_mse  ', self.metrics['mkt_mse'] / self.metrics['n'], self.metrics['n'])
            print('ba corr  ', np.sum(np.round(self.corr_me) == np.round(self.corr_mkt)) / len(self.corr_mkt))
            print('corr r   ', r)
            print('corr r2  ', r_squared)

    def place_bets(self, summary: pd.DataFrame, opps: pd.DataFrame, inc: tuple[pd.DataFrame, pd.DataFrame]):
        games_increment, players_increment = inc

        self.print_metrics()

        min_bet = summary.iloc[0]['Min_bet']
        max_bet = summary.iloc[0]['Max_bet']

        bets = pd.DataFrame(data=np.zeros((len(opps), 2)), columns=['BetH', 'BetA'], index=opps.index)

        for i in games_increment.index:
            current = games_increment.loc[i]

            home_win = current['H']
            odds_home = current['OddsH']
            odds_away = current['OddsA']
            overround = 1 / odds_home + 1 / odds_away
            mkt_pred = 1 / odds_home / overround

            str_i = str(i)

            if str_i in predictions:
                # self.past_pred.append([0.0, home_win])
                self.past_pred.append([predictions[str_i], home_win])

            if i in self.prediction_map:
                if self.prediction_map[i] == 0.5:
                    self.metrics['my_ba'] += 0.5
                elif (self.prediction_map[i] > 0.5) == home_win:
                    self.metrics['my_ba'] += 1

                if mkt_pred == 0.5:
                    self.metrics['mkt_ba'] += 0.5
                elif (mkt_pred > 0.5) == home_win:
                    self.metrics['mkt_ba'] += 1

                self.metrics['my_mse'] += (self.prediction_map[i] - home_win) ** 2
                self.metrics['mkt_mse'] += (mkt_pred - home_win) ** 2
                self.metrics['n'] += 1

                self.corr_me.append(self.prediction_map[i])
                self.corr_mkt.append(mkt_pred)

        for i in opps.index:
            current = opps.loc[i]

            str_i = str(i)

            if current['Date'] == summary.iloc[0]['Date'] and str_i in predictions:
                if len(self.past_pred) >= 1500:
                    np_array = np.array(self.past_pred)
                    sample_weights = np.exp(-0.0003 * np.arange(len(self.past_pred)))
                    lr = LogisticRegression(max_iter=10000)
                    lr.fit(np_array[:, :-1], np_array[:, -1], sample_weight=sample_weights[::-1])
                    pred = lr.predict_proba(np.array([[predictions[str_i]]]))[0, 1]
                    # pred = lr.predict_proba(np.array([[0.0]]))[0, 1]

                    self.prediction_map[i] = pred

                    odds_home = current['OddsH']
                    odds_away = current['OddsA']

                    min_home_odds = (1 / pred - 1) * 1.5 + 1 + 0.1
                    min_away_odds = (1 / (1 - pred) - 1) * 1.5 + 1 + 0.1

                    if odds_home >= min_home_odds:
                        bets.at[i, 'BetH'] = min_bet

                        self.exp_pnl += pred * odds_home - 1
                        self.bet_volume += min_bet
                        self.bet_count += 1
                        self.bet_sum_odds += odds_home

                    if odds_away >= min_away_odds:
                        bets.at[i, 'BetA'] = min_bet

                        self.exp_pnl += (1 - pred) * odds_away - 1
                        self.bet_volume += min_bet
                        self.bet_count += 1
                        self.bet_sum_odds += odds_away

        return bets
