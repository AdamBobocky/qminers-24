import math
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.linear_model import LogisticRegression

def get_player_performance(row):
    return np.sum(row[['ORB', 'DRB', 'AST', 'STL', 'BLK', 'FGA', 'FTA', 'TOV', 'PF']].to_numpy() / row['MIN'] * np.array([-0.72544444, 0.56819659, 0.3809517, 0.01808557, 0.47349012, -0.05964727, 0.26915364, -1.27520841, 0.01561063]))

class Model:
    def __init__(self):
        self.factor_map = {}
        self.prediction_map = {}
        self.past_pred = []
        self.corr_me = []
        self.corr_mkt = []

        self.team_rosters = {}

        self.player_scores = {}

        self.metrics = {
            'my_mse': 0,
            'mkt_mse': 0,
            'n': 0
        }

        self.bet_opps = 0
        self.exp_pnl = 0
        self.bet_volume = 0
        self.bet_count = 0
        self.bet_sum_odds = 0

    def get_player_score(self, pid):
        data = self.player_scores[pid]

        total_weighted_sum = 0
        total_weight = 0

        for i, value in enumerate(reversed(data)):
            weight = 0.93 ** i
            total_weighted_sum += value * weight
            total_weight += weight

        return total_weighted_sum / total_weight if total_weight > 3 else 0

    def predict(self, season, home_id, away_id):
        if season not in self.team_rosters:
            return -9999
        if home_id not in self.team_rosters[season] or len(self.team_rosters[season][home_id]) < 5:
            return -9999
        if away_id not in self.team_rosters[season] or len(self.team_rosters[season][away_id]) < 5:
            return -9999

        home_rosters = self.team_rosters[season][home_id][-5:]
        away_rosters = self.team_rosters[season][away_id][-5:]

        home_factor = 0
        away_factor = 0
        home_minutes = 0
        away_minutes = 0

        for roster in home_rosters:
            for pid, mins in roster:
                home_factor += self.get_player_score(pid) * int(mins)
                home_minutes += int(mins)

        for roster in away_rosters:
            for pid, mins in roster:
                away_factor += self.get_player_score(pid) * int(mins)
                away_minutes += int(mins)

        if home_minutes < 500 or away_minutes < 500:
            return -9999

        print('\nx:', home_factor / home_minutes - away_factor / away_minutes)

        return home_factor / home_minutes - away_factor / away_minutes

    def place_bets(self, summary: pd.DataFrame, opps: pd.DataFrame, inc: tuple[pd.DataFrame, pd.DataFrame]):
        games_increment, players_increment = inc

        if self.bet_count > 0:
            print('\nOpps:', self.bet_opps, 'Bets:', self.bet_count, 'Volume:', self.bet_volume, 'Avg odds:', self.bet_sum_odds / self.bet_count, 'Exp avg P&L:', self.exp_pnl / self.bet_count)

        if self.metrics['n'] > 0:
            r = np.corrcoef(self.corr_me, self.corr_mkt)[0, 1]
            r_squared = r ** 2

            print('my_mse   ', self.metrics['my_mse'] / self.metrics['n'], self.metrics['n'])
            print('mkt_mse  ', self.metrics['mkt_mse'] / self.metrics['n'], self.metrics['n'])
            print('corr r   ', r)
            print('corr r2  ', r_squared)

        for i in players_increment.index:
            current = players_increment.loc[i]

            PID = current['Player']

            if PID not in self.player_scores:
                self.player_scores[PID] = []

            if current['MIN'] >= 3:
                performance = get_player_performance(current)
                if not math.isnan(performance):
                    self.player_scores[PID].append(performance)

        for i in games_increment.index:
            current = games_increment.loc[i]

            season = current['Season']
            home_id = current['HID']
            away_id = current['AID']
            home_score = current['HSC']
            away_score = current['ASC']
            home_win = current['H']
            odds_home = current['OddsH']
            odds_away = current['OddsA']
            overround = 1 / odds_home + 1 / odds_away
            mkt_pred = 1 / odds_home / overround

            game_players = players_increment[(players_increment['Game'] == i) & (players_increment['MIN'] >= 3)]

            home_players = game_players[game_players['Team'] == current['HID']]
            away_players = game_players[game_players['Team'] == current['AID']]

            if season not in self.team_rosters:
                self.team_rosters[season] = {}

            if home_id not in self.team_rosters[season]:
                self.team_rosters[season][home_id] = []

            if away_id not in self.team_rosters[season]:
                self.team_rosters[season][away_id] = []

            self.team_rosters[season][home_id].append([[x['Player'], x['MIN']] for _, x in home_players.iterrows()])
            self.team_rosters[season][away_id].append([[x['Player'], x['MIN']] for _, x in away_players.iterrows()])

            if i in self.factor_map and self.factor_map[i] != -9999:
                self.past_pred.append([self.factor_map[i], mkt_pred, home_win])

            if i in self.prediction_map:
                self.metrics['my_mse'] += (self.prediction_map[i] - home_win) ** 2
                self.metrics['mkt_mse'] += (mkt_pred - home_win) ** 2
                self.metrics['n'] += 1

                self.corr_me.append(self.prediction_map[i])
                self.corr_mkt.append(mkt_pred)

        min_bet = summary.iloc[0]["Min_bet"]

        bets = pd.DataFrame(data=np.zeros((len(opps), 2)), columns=["BetH", "BetA"], index=opps.index)

        for i in opps.index:
            current = opps.loc[i]

            odds_home = current['OddsH']
            odds_away = current['OddsA']
            overround = 1 / odds_home + 1 / odds_away
            mkt_pred = 1 / odds_home / overround

            if current['Date'] == summary.iloc[0]['Date']:
                factor = self.predict(current['Season'], current['HID'], current['AID'])

                self.factor_map[i] = factor

                if len(self.past_pred) >= 500 and factor != -9999:
                    np_array = np.array(self.past_pred[-2000:])

                    self.bet_opps += 1

                    lr = LogisticRegression()
                    lr.fit(np_array[:, :-1], np_array[:, -1])

                    # print('\nModel intercept: ', lr.intercept_, 'coefficients:', lr.coef_)

                    pred = lr.predict_proba(np.array([factor, mkt_pred]).reshape(1, -1))[0, 1]

                    self.prediction_map[i] = pred

                    min_home_odds = (1 / pred - 1) * 1.15 + 1 + 0.05
                    min_away_odds = (1 / (1 - pred) - 1) * 1.15 + 1 + 0.05

                    if odds_home >= min_home_odds and odds_home < 4:
                        bets.at[i, 'BetH'] = min_bet

                        self.exp_pnl += pred * odds_home - 1
                        self.bet_volume += min_bet
                        self.bet_count += 1
                        self.bet_sum_odds += odds_home

                    if odds_away >= min_away_odds and odds_away < 4:
                        bets.at[i, 'BetA'] = min_bet

                        self.exp_pnl += (1 - pred) * odds_away - 1
                        self.bet_volume += min_bet
                        self.bet_count += 1
                        self.bet_sum_odds += odds_away

        return bets
