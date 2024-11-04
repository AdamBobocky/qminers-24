from datetime import datetime
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

class GradientDescent:
    def __init__(self, num_teams, learning_rate=0.01, prior_alpha=0.1, prior_beta=0.1, monthly_decay=0.8):
        # Hyperparameters
        self.learning_rate = learning_rate
        self.prior_alpha = prior_alpha
        self.prior_beta = prior_beta
        self.monthly_decay = monthly_decay

        # Games storage
        self.games = np.empty((0, 5), int)

        # Tuned params
        self.team_mus = np.random.randn(num_teams) * 0.01
        self.team_sigmas = np.random.randn(num_teams) * 0.01
        self.team_advantages = np.random.randn(num_teams) * 0.01

    def _get_average_home_advantage(self):
        weights = self._get_time_weights()
        advantages = self.games[:, 3] - self.games[:, 4]

        return np.sum(advantages * weights) / np.sum(weights)

    def _gradients(self, average_home_advantage):
        weights = self._get_time_weights()

        expectations = average_home_advantage + self.team_advantages[self.games[:, 1]] + self.team_mus[self.games[:, 1]] - self.team_mus[self.games[:, 2]]

        realities = self.games[:, 3] - self.games[:, 4]

        mus_adjustments_home = (realities - expectations) * weights
        mus_adjustments_away = -(realities - expectations) * weights

        advantage_adjustments_home = (realities - expectations) * np.sqrt(weights)

        grad_team_mus = np.zeros_like(self.team_mus)
        grad_team_advantages = np.zeros_like(self.team_advantages)

        np.add.at(grad_team_mus, self.games[:, 1], mus_adjustments_home)
        np.add.at(grad_team_mus, self.games[:, 2], mus_adjustments_away)

        np.add.at(grad_team_advantages, self.games[:, 1], advantage_adjustments_home)

        return grad_team_mus, grad_team_advantages

    def _get_time_weights(self):
        last_ts = self.games[-1, 0]

        return self.monthly_decay ** (np.abs(self.games[:, 0] - last_ts) / 30 / 24 / 60 / 60 / 1000)

    def _time_weight(self, timestamp):
        delta_months = abs(timestamp - self.games[-1][0]) / 30 / 24 / 60 / 60 / 1000

        return pow(self.monthly_decay, delta_months)

    def _calculate_objective(self, average_home_advantage):
        objective = 0

        weights = self._get_time_weights()

        expectations = average_home_advantage + self.team_advantages[self.games[:, 1]] + self.team_mus[self.games[:, 1]] - self.team_mus[self.games[:, 2]]

        realities = self.games[:, 3] - self.games[:, 4]

        return -np.sum((realities - expectations) ** 2 * weights)

    def add_game(self, timestamp, team_home, team_away, score_home, score_away):
        self.games = np.vstack([self.games, np.array([timestamp, team_home, team_away, score_home, score_away])])

    def fit(self):
        average_home_advantage = self._get_average_home_advantage()
        best_objective = self._calculate_objective(average_home_advantage)
        games_count = len(self.games)
        countdown = 10
        while countdown > 0:
            countdown -= 1

            grad_team_mus, grad_team_advantages = self._gradients(average_home_advantage)

            self.team_mus += self.learning_rate * grad_team_mus
            self.team_advantages += self.learning_rate * grad_team_advantages

            new_objective = self._calculate_objective(average_home_advantage) / games_count

            # print(new_objective / games_count, end='\r')
            if new_objective > best_objective + 0.000001:
                best_objective = new_objective
                countdown = 10

        # print('')

    def predict(self, team_home, team_away):
        return self._get_average_home_advantage() + self.team_advantages[team_home] + self.team_mus[team_home] - self.team_mus[team_away]

class Model:
    def __init__(self):
        self.model = GradientDescent(30)
        self.factor_map = {}
        self.prediction_map = {}
        self.my_team_id = {}
        self.num_teams = 0
        self.countdown = 2000
        self.past_pred = []
        self.corr_me = []
        self.corr_mkt = []

        self.metrics = {
            'my_mse': 0,
            'mkt_mse': 0,
            'n': 0
        }

        self.bet_opps = 0
        self.bet_volume = 0
        self.bet_count = 0
        self.bet_sum_odds = 0

    def place_bets(self, summary: pd.DataFrame, opps: pd.DataFrame, inc: tuple[pd.DataFrame, pd.DataFrame]):
        games_increment, players_increment = inc

        if self.bet_count > 0:
            print()
            print('Opps:', self.bet_opps, 'Bets:', self.bet_count, 'Volume:', self.bet_volume, 'Avg odds:', self.bet_sum_odds / self.bet_count)

        if self.metrics['n'] > 0:
            r = np.corrcoef(self.corr_me, self.corr_mkt)[0, 1]
            r_squared = r ** 2

            print('')
            print('my_mse   ', self.metrics['my_mse'] / self.metrics['n'], self.metrics['n'])
            print('mkt_mse  ', self.metrics['mkt_mse'] / self.metrics['n'], self.metrics['n'])
            print('corr r   ', r)
            print('corr r2  ', r_squared)

        for i in games_increment.index:
            current = games_increment.loc[i]

            timestamp = int(current['Date'].timestamp() * 1000)
            home_id = current['HID']
            away_id = current['AID']
            home_score = current['HSC']
            away_score = current['ASC']
            home_win = current['H']
            odds_home = current['OddsH']
            odds_away = current['OddsA']
            overround = 1 / odds_home + 1 / odds_away
            mkt_pred = 1 / odds_home / overround

            if home_id not in self.my_team_id:
                self.my_team_id[home_id] = self.num_teams
                self.num_teams += 1

            if away_id not in self.my_team_id:
                self.my_team_id[away_id] = self.num_teams
                self.num_teams += 1

            if i in self.factor_map:
                factor = self.factor_map[i]
                self.past_pred.append([factor, home_win])

            if i in self.prediction_map:
                self.metrics['my_mse'] += (self.prediction_map[i] - home_win) ** 2
                self.metrics['mkt_mse'] += (mkt_pred - home_win) ** 2
                self.metrics['n'] += 1

                self.corr_me.append(self.prediction_map[i])
                self.corr_mkt.append(mkt_pred)

            self.countdown -= 1
            self.model.add_game(timestamp, self.my_team_id[home_id], self.my_team_id[away_id], home_score, away_score)

        min_bet = summary.iloc[0]["Min_bet"]

        bets = pd.DataFrame(data=np.zeros((len(opps), 2)), columns=["BetH", "BetA"], index=opps.index)

        if self.countdown <= 0:
            self.model.fit()

            for i in opps.index:
                current = opps.loc[i]

                if current['Date'] == summary.iloc[0]['Date'] and current['HID'] in self.my_team_id and current['AID'] in self.my_team_id:
                    my_factor = self.model.predict(self.my_team_id[current['HID']], self.my_team_id[current['AID']])

                    self.factor_map[i] = my_factor

                    if len(self.past_pred) >= 500:
                        np_array = np.array(self.past_pred[-2000:])

                        self.bet_opps += 1

                        lr = LogisticRegression()
                        lr.fit(np_array[:, :-1], np_array[:, -1])

                        pred = lr.predict_proba(np.array([my_factor]).reshape(1, -1))[0, 1]

                        self.prediction_map[i] = pred

                        odds_home = current['OddsH']
                        odds_away = current['OddsA']

                        if pred * odds_home > 1.2 and odds_home < 5:
                            bets.at[i, 'BetH'] = min_bet

                            self.bet_volume += min_bet
                            self.bet_count += 1
                            self.bet_sum_odds += odds_home

                        if (1 - pred) * odds_away > 1.2 and odds_away < 5:
                            bets.at[i, 'BetA'] = min_bet

                            self.bet_volume += min_bet
                            self.bet_count += 1
                            self.bet_sum_odds += odds_away

        return bets
