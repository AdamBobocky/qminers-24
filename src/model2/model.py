import math
import numpy as np
import pandas as pd
import json
from datetime import datetime
from sklearn.linear_model import LogisticRegression

# Let's assume that difference in scores is normally distributed
# And that team ratings are also normally distributed

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def log_likelihood(delta, game_sigma):
    # Log likelihood:
    #   ln(1 / (game_sigma * 2.50662828) * e ** (-0.5 * (delta / game_sigma) ** 2))

    return 1 / (game_sigma * 2.50662828) * np.exp(-0.5 * (delta / game_sigma) ** 2)
    # return np.log(1 / (game_sigma * 2.50662828) * np.exp(-0.5 * (delta / game_sigma) ** 2))

def log_likelihood_derivative_wrt_my_rating(delta, game_sigma):
    return delta / game_sigma ** 2

def log_likelihood_derivative_wrt_delta(delta, game_sigma):
    # Log likelihood derivative wrt delta:
    #   -delta / game_sigma ** 2

    return -delta / game_sigma ** 2

def log_likelihood_derivative_wrt_sigma(delta, game_sigma):
    # Log likelihood derivative wrt sigma:
    #   -(game_sigma ** 2 - delta ** 2) / game_sigma ** 3

    return -(game_sigma ** 2 - delta ** 2) / game_sigma ** 3

class GradientDescent:
    def __init__(self, num_teams, learning_rate=0.03, monthly_decay=0.9, season_reset_mult=0.2):
        # Hyperparameters
        self.learning_rate = learning_rate
        self.monthly_decay = monthly_decay
        self.season_reset_mult = season_reset_mult

        # Games storage
        self.games = np.empty((0, 5), int)

        # Tuned params
        self.home_advantage = 5
        self.sigma = 12 # Parameter for game variance outside of team uncertainty
        self.team_mus = np.zeros(num_teams)
        self.team_sigmas = np.ones(num_teams) * 128 / 3

    def _gradients(self):
        weights = self._get_time_weights()

        home_ratings = self.team_mus[self.games[:, 1]]
        away_ratings = self.team_mus[self.games[:, 2]]

        expectations_home = self.home_advantage + home_ratings - away_ratings

        realities_home = self.games[:, 3] - self.games[:, 4]
        realities_away = self.games[:, 4] - self.games[:, 3]

        game_sigmas = np.sqrt(self.team_sigmas[self.games[:, 1]] ** 2 + self.team_sigmas[self.games[:, 2]] ** 2 + self.sigma ** 2)

        mus_adjustments_home = log_likelihood_derivative_wrt_my_rating(realities_home + away_ratings - self.home_advantage - home_ratings, game_sigmas) * weights
        mus_adjustments_away = log_likelihood_derivative_wrt_my_rating(realities_away + home_ratings + self.home_advantage - away_ratings, game_sigmas) * weights
        grad_sigma = log_likelihood_derivative_wrt_sigma(realities_home - expectations_home, game_sigmas) * weights
        grad_home_advantage = log_likelihood_derivative_wrt_delta(realities_home - expectations_home, game_sigmas) * weights

        grad_team_mus = np.zeros_like(self.team_mus)

        np.add.at(grad_team_mus, self.games[:, 1], mus_adjustments_home)
        np.add.at(grad_team_mus, self.games[:, 2], mus_adjustments_away)

        return grad_team_mus, np.mean(grad_sigma), np.mean(grad_home_advantage)

    def _get_time_weights(self):
        last_ts = self.games[-1, 0]

        return self.monthly_decay ** (np.abs(self.games[:, 0] - last_ts) / 30 / 24 / 60 / 60 / 1000)

    def _time_weight(self, timestamp):
        delta_months = abs(timestamp - self.games[-1][0]) / 30 / 24 / 60 / 60 / 1000

        return pow(self.monthly_decay, delta_months)

    def _calculate_objective(self):
        weights = self._get_time_weights()

        home_ratings = self.team_mus[self.games[:, 1]]
        away_ratings = self.team_mus[self.games[:, 2]]

        expectations_home = self.home_advantage + home_ratings - away_ratings

        realities_home = self.games[:, 3] - self.games[:, 4]
        realities_away = self.games[:, 4] - self.games[:, 3]

        game_sigmas = np.sqrt(self.team_sigmas[self.games[:, 1]] ** 2 + self.team_sigmas[self.games[:, 2]] ** 2 + self.sigma ** 2)

        home_objective = np.sum(log_likelihood(realities_home + away_ratings - self.home_advantage - home_ratings, game_sigmas) * weights)
        away_objective = np.sum(log_likelihood(realities_away + home_ratings + self.home_advantage - away_ratings, game_sigmas) * weights)
        sigma_objective = np.sum(log_likelihood(realities_home - expectations_home, game_sigmas) * weights) * 0.1

        return home_objective + away_objective + sigma_objective

    def add_game(self, timestamp, team_home, team_away, score_home, score_away):
        self.games = np.vstack([self.games, np.array([timestamp, team_home, team_away, score_home, score_away])])
        self.games = self.games[-6000:]

        # Reduce sigma
        self.team_sigmas[team_home] = 1 / math.sqrt(1 / self.team_sigmas[team_home] ** 2 + 1 / self.sigma ** 2)
        self.team_sigmas[team_away] = 1 / math.sqrt(1 / self.team_sigmas[team_away] ** 2 + 1 / self.sigma ** 2)

    def new_season(self):
        self.team_sigmas = np.ones_like(self.team_sigmas) * (128 * self.season_reset_mult)

    def fit(self):
        best_objective = self._calculate_objective()
        games_count = len(self.games)
        countdown = 50
        while countdown > 0:
            countdown -= 1

            grad_team_mus, grad_sigma, grad_home_advantage = self._gradients()

            self.team_mus += self.learning_rate * grad_team_mus
            self.sigma += self.learning_rate * grad_sigma
            self.home_advantage -= self.learning_rate * grad_home_advantage

            new_objective = self._calculate_objective() / games_count

            print(new_objective, end='\r')
            if new_objective > best_objective + 0.000001:
                best_objective = new_objective
                countdown = 50

    def predict(self, team_home, team_away):
        game_exp = self.home_advantage + self.team_mus[team_home] - self.team_mus[team_away]
        game_sigma = math.sqrt(self.team_sigmas[team_home] ** 2 + self.team_sigmas[team_away] ** 2 + self.sigma ** 2)

        # return sigmoid(game_exp / game_sigma * 1.65), game_exp
        return game_exp / game_sigma, game_exp

class Model:
    def __init__(self, monthly_decay, season_reset_mult):
        self.model = GradientDescent(30, 0.03, monthly_decay, season_reset_mult)
        self.exp_map = {}
        self.prediction_map = {}
        self.src_prediction_map = {}
        self.my_team_id = {}
        self.num_teams = 0
        self.countdown = 3500
        self.past_pred = []
        self.pred_list = []
        self.corr_me = []
        self.corr_mkt = []
        self.last_season = -1

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

    def place_bets(self, summary: pd.DataFrame, opps: pd.DataFrame, inc: tuple[pd.DataFrame, pd.DataFrame]):
        games_increment, players_increment = inc

        with open('src/model2/data.json', 'w') as json_file:
            json.dump({
                'team_ratings': self.model.team_mus.tolist(),
                'corr_me': self.corr_me,
                'corr_mkt': self.corr_mkt,
                'prediction_map': self.prediction_map,
                'pred_list': self.pred_list
            }, json_file, indent=2)

        print(f'\nParams: {self.model.home_advantage} {self.model.sigma}')

        if self.bet_count > 0:
            print()
            print('Opps:', self.bet_opps, 'Bets:', self.bet_count, 'Volume:', self.bet_volume, 'Avg odds:', self.bet_sum_odds / self.bet_count, 'Exp avg P&L:', self.exp_pnl / self.bet_count)

        if self.metrics['n'] > 0:
            r = np.corrcoef(self.corr_me, self.corr_mkt)[0, 1]
            r_squared = r ** 2

            print('')
            print('my_ba    ', self.metrics['my_ba'] / self.metrics['n'], self.metrics['n'])
            print('mkt_ba   ', self.metrics['mkt_ba'] / self.metrics['n'], self.metrics['n'])
            print('my_mse   ', self.metrics['my_mse'] / self.metrics['n'], self.metrics['n'])
            print('mkt_mse  ', self.metrics['mkt_mse'] / self.metrics['n'], self.metrics['n'])
            print('ba corr  ', np.sum(np.round(self.corr_me) == np.round(self.corr_mkt)) / len(self.corr_mkt))
            print('corr r   ', r)
            print('corr r2  ', r_squared)

            with open('mse.json', 'w') as json_file:
                json.dump({
                    'mse': self.metrics['my_mse'] / self.metrics['n']
                }, json_file, indent=2)

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

            if i in self.src_prediction_map:
                src_pred = self.src_prediction_map[i]
                self.past_pred.append([src_pred, home_win])

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

                self.pred_list.append({
                    'index': str(i),
                    'playoff': int(current['POFF']),
                    'date': str(current['Date']),
                    'season': int(current['Season']),
                    'score': int(home_score - away_score),
                    'my_exp': self.exp_map[i],
                    'my_pred': self.prediction_map[i],
                    'mkt_pred': mkt_pred,
                    'odds_home': float(odds_home),
                    'odds_away': float(odds_away),
                    'outcome': int(home_win)
                })

            self.countdown -= 1
            self.model.add_game(timestamp, self.my_team_id[home_id], self.my_team_id[away_id], home_score, away_score)

        min_bet = summary.iloc[0]['Min_bet']
        max_bet = summary.iloc[0]['Max_bet']

        bets = pd.DataFrame(data=np.zeros((len(opps), 2)), columns=['BetH', 'BetA'], index=opps.index)

        if self.countdown <= 0:
            for i in opps.index:
                current = opps.loc[i]

                if self.last_season != current['Season']:
                    self.last_season = current['Season']

                    self.model.new_season()

            self.model.fit()

            for i in opps.index:
                current = opps.loc[i]

                if current['Date'] == summary.iloc[0]['Date'] and current['HID'] in self.my_team_id and current['AID'] in self.my_team_id:
                    src_pred, exp = self.model.predict(self.my_team_id[current['HID']], self.my_team_id[current['AID']])

                    self.src_prediction_map[i] = src_pred
                    self.exp_map[i] = exp

                    if len(self.past_pred) >= 500:
                        np_array = np.array(self.past_pred[-2000:])

                        self.bet_opps += 1

                        lr = LogisticRegression()
                        # lr = LogisticRegression(fit_intercept=False)
                        lr.fit(np_array[:, :-1], np_array[:, -1])

                        # print('\nModel intercept: ', lr.intercept_, 'coefficients:', lr.coef_)

                        pred = lr.predict_proba(np.array([src_pred]).reshape(1, -1))[0, 1]

                        # self.prediction_map[i] = pred

                        self.prediction_map[i] = pred

                        odds_home = current['OddsH']
                        odds_away = current['OddsA']

                        min_home_odds = (1 / pred - 1) * 1.3 + 1 + 0.04
                        min_away_odds = (1 / (1 - pred) - 1) * 1.3 + 1 + 0.04

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
