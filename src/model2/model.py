import math
import numpy as np
import pandas as pd
# import json
from datetime import datetime
from sklearn.linear_model import LogisticRegression
from collections import defaultdict

def inverse_sigmoid(x):
    return math.log(x / (1 - x))

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
    def __init__(self, num_teams, learning_rate=0.01, monthly_decay=0.9, season_reset_mult=0.2):
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
        games_count = len(self.games)
        best_objective = self._calculate_objective() / games_count
        best_state = [self.team_mus, self.sigma, self.home_advantage]
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
                best_state = [self.team_mus, self.sigma, self.home_advantage]
                countdown = 50

        self.team_mus, self.sigma, self.home_advantage = best_state

    def predict(self, team_home, team_away):
        game_exp = self.home_advantage + self.team_mus[team_home] - self.team_mus[team_away]
        game_sigma = math.sqrt(self.team_sigmas[team_home] ** 2 + self.team_sigmas[team_away] ** 2 + self.sigma ** 2)

        return game_exp / game_sigma

class FourFactor:
    def __init__(self):
        self.team_stats_average = defaultdict(list)
        self.opponent_stats_average = defaultdict(list)

    def _get_stats(self, date, stats):
        totals = {
            'FieldGoalsMade': 0,
            '3PFieldGoalsMade': 0,
            'FieldGoalAttempts': 0,
            'Turnovers': 0,
            'OffensiveRebounds': 0,
            'OpponentsDefensiveRebounds': 0,
            'FreeThrowAttempts': 0,
            'Score': 0,
            'Win': 0,
            'Weight': 0
        }

        # Iterate over each dictionary in the list
        for stat in stats:
            weight = 0.994 ** abs((date - stat['Date']).days)

            # Multiply each relevant field by the weight and add to totals
            totals['FieldGoalsMade'] += stat['FieldGoalsMade'] * weight
            totals['3PFieldGoalsMade'] += stat['3PFieldGoalsMade'] * weight
            totals['FieldGoalAttempts'] += stat['FieldGoalAttempts'] * weight
            totals['Turnovers'] += stat['Turnovers'] * weight
            totals['OffensiveRebounds'] += stat['OffensiveRebounds'] * weight
            totals['OpponentsDefensiveRebounds'] += stat['OpponentsDefensiveRebounds'] * weight
            totals['FreeThrowAttempts'] += stat['FreeThrowAttempts'] * weight
            totals['Score'] += stat['Score'] * weight
            totals['Win'] += stat['Win'] * weight
            totals['Weight'] += weight

        return totals

    def add_game(self, current):
        self.team_stats_average[current['HID']].append({
            'Date': current['Date'],
            'FieldGoalsMade': current['HFGM'],
            '3PFieldGoalsMade': current['HFG3M'],
            'FieldGoalAttempts': current['HFGA'],
            'Turnovers': current['HTOV'],
            'OffensiveRebounds': current['HORB'],
            'OpponentsDefensiveRebounds': current['ADRB'],
            'FreeThrowAttempts': current['HFTA'],
            'Score': current['HSC'],
            'Win': current['H']
        })
        self.team_stats_average[current['AID']].append({
            'Date': current['Date'],
            'FieldGoalsMade': current['AFGM'],
            '3PFieldGoalsMade': current['AFG3M'],
            'FieldGoalAttempts': current['AFGA'],
            'Turnovers': current['ATOV'],
            'OffensiveRebounds': current['AORB'],
            'OpponentsDefensiveRebounds': current['ADRB'],
            'FreeThrowAttempts': current['AFTA'],
            'Score': current['ASC'],
            'Win': current['A']
        })
        # Opponent
        self.opponent_stats_average[current['AID']].append({
            'Date': current['Date'],
            'FieldGoalsMade': current['HFGM'],
            '3PFieldGoalsMade': current['HFG3M'],
            'FieldGoalAttempts': current['HFGA'],
            'Turnovers': current['HTOV'],
            'OffensiveRebounds': current['HORB'],
            'OpponentsDefensiveRebounds': current['ADRB'],
            'FreeThrowAttempts': current['HFTA'],
            'Score': current['HSC'],
            'Win': current['H']
        })
        self.opponent_stats_average[current['HID']].append({
            'Date': current['Date'],
            'FieldGoalsMade': current['AFGM'],
            '3PFieldGoalsMade': current['AFG3M'],
            'FieldGoalAttempts': current['AFGA'],
            'Turnovers': current['ATOV'],
            'OffensiveRebounds': current['AORB'],
            'OpponentsDefensiveRebounds': current['ADRB'],
            'FreeThrowAttempts': current['AFTA'],
            'Score': current['ASC'],
            'Win': current['A']
        })

    def get_input_data(self, date, team_id):
        if len(self.team_stats_average[team_id]) <= 5:
            return None

        stats = self._get_stats(date, self.team_stats_average[team_id])
        opp_stats = self._get_stats(date, self.opponent_stats_average[team_id])

        return [
            (stats['FieldGoalsMade'] + 0.5 * stats['3PFieldGoalsMade']) / stats['FieldGoalAttempts'],
            stats['Turnovers'] / (stats['FieldGoalAttempts'] + 0.44 * stats['FreeThrowAttempts'] + stats['Turnovers']),
            stats['OffensiveRebounds'] / (stats['OffensiveRebounds'] + stats['OpponentsDefensiveRebounds']),
            stats['FreeThrowAttempts'] / stats['FieldGoalAttempts'],
            stats['Score'] / stats['Weight'],
            (opp_stats['FieldGoalsMade'] + 0.5 * opp_stats['3PFieldGoalsMade']) / opp_stats['FieldGoalAttempts'],
            opp_stats['Turnovers'] / (opp_stats['FieldGoalAttempts'] + 0.44 * opp_stats['FreeThrowAttempts'] + opp_stats['Turnovers']),
            opp_stats['OffensiveRebounds'] / (opp_stats['OffensiveRebounds'] + opp_stats['OpponentsDefensiveRebounds']),
            opp_stats['FreeThrowAttempts'] / opp_stats['FieldGoalAttempts'],
            opp_stats['Score'] / opp_stats['Weight']
        ]

class Elo:
    def __init__(self, k_factor = 0.15, home_factor = 0.5):
        self.k_factor = k_factor
        self.home_factor = home_factor

        self.elo_map = defaultdict(float)

    def predict(self, home_id, away_id):
        return sigmoid(self.elo_map[home_id] - self.elo_map[away_id] + self.home_factor)

    def add_game(self, current):
        home_id = current['HID']
        away_id = current['AID']

        elo_prediction = self.predict(home_id, away_id)

        self.elo_map[home_id] += self.k_factor * (current['H'] - elo_prediction)
        self.elo_map[away_id] += self.k_factor * (current['A'] - (1 - elo_prediction))

class Model:
    def __init__(self, monthly_decay=0.75, season_reset_mult=0.8):
        self.gd_model = GradientDescent(30, 0.03, monthly_decay, season_reset_mult)
        self.ff_model = FourFactor()
        self.el_model = Elo()

        self.prediction_map = {}
        self.my_team_id = {}
        self.countdown = 3500
        self.past_pred = []
        self.pred_list = []
        self.corr_me = []
        self.corr_mkt = []
        self.last_season = -1
        self.season_start = 0
        self.lr = None
        self.lr_retrain = 0

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

    def get_input_features(self, home_id, away_id, date):
        if home_id not in self.my_team_id:
            return None
        if away_id not in self.my_team_id:
            return None

        home_ff = self.ff_model.get_input_data(date, home_id)
        away_ff = self.ff_model.get_input_data(date, away_id)

        if home_ff is None or away_ff is None:
            return None

        return [
            self.el_model.predict(home_id, away_id),
            self.gd_model.predict(self.my_team_id[home_id], self.my_team_id[away_id]),
            *home_ff,
            *away_ff
        ]

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

        last_fit = None

        self.print_metrics()

        # with open('src/model2/data.json', 'w') as json_file:
        #     json.dump(self.pred_list, json_file, indent=2)

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
                self.my_team_id[home_id] = len(self.my_team_id)

            if away_id not in self.my_team_id:
                self.my_team_id[away_id] = len(self.my_team_id)

            if self.last_season != current['Season']:
                self.last_season = current['Season']
                self.season_start = current['Date']
                self.gd_model.new_season()

            if self.countdown <= 0:
                if last_fit is None or last_fit != current['Date']:
                    last_fit = current['Date']
                    self.gd_model.fit()

                input_arr = self.get_input_features(home_id, away_id, current['Date'])
                if input_arr is not None:
                    self.lr_retrain -= 1
                    self.past_pred.append([*input_arr, home_win])

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
                    'neutral': int(current['N']),
                    'playoff': int(current['POFF']),
                    'date': str(current['Date']),
                    'season': int(current['Season']),
                    'score': int(home_score - away_score),
                    'my_pred': self.prediction_map[i],
                    'mkt_pred': mkt_pred,
                    'odds_home': float(odds_home),
                    'odds_away': float(odds_away),
                    'outcome': int(home_win)
                })

            self.countdown -= 1
            self.el_model.add_game(current)
            self.ff_model.add_game(current)
            self.gd_model.add_game(timestamp, self.my_team_id[home_id], self.my_team_id[away_id], home_score, away_score)

        min_bet = summary.iloc[0]['Min_bet']
        max_bet = summary.iloc[0]['Max_bet']

        bets = pd.DataFrame(data=np.zeros((len(opps), 2)), columns=['BetH', 'BetA'], index=opps.index)

        if self.countdown <= 0:
            self.gd_model.fit()

            for i in opps.index:
                current = opps.loc[i]

                date = current['Date']
                home_id = current['HID']
                away_id = current['AID']
                week = (date - self.season_start).days / 7
                conf = 0.00838 * week + 0.826

                if date == summary.iloc[0]['Date'] and self.last_season == current['Season']:
                    input_arr = self.get_input_features(home_id, away_id, date)

                    if input_arr is not None and len(self.past_pred) >= 1500:
                        if self.lr_retrain <= 0:
                            self.lr_retrain += 200
                            np_array = np.array(self.past_pred)
                            sample_weights = np.exp(-0.0003 * np.arange(len(self.past_pred)))
                            self.lr = LogisticRegression(max_iter=10000)
                            self.lr.fit(np_array[:, :-1], np_array[:, -1], sample_weight=sample_weights[::-1])

                        self.bet_opps += 1

                        pred = self.lr.predict_proba(np.array([input_arr]))[0, 1]

                        pred = sigmoid(inverse_sigmoid(pred) * conf)

                        self.prediction_map[i] = pred

                        odds_home = current['OddsH']
                        odds_away = current['OddsA']

                        min_home_odds = (1 / pred - 1) * 1.25 + 1 + 0.00
                        min_away_odds = (1 / (1 - pred) - 1) * 1.25 + 1 + 0.00

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
