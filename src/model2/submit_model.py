import math
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.linear_model import LogisticRegression
from collections import defaultdict
import warnings

warnings.filterwarnings('ignore')

def inverse_sigmoid(x):
    return math.log(x / (1 - x))

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def x1(delta, game_sigma):
    return 1 / (game_sigma * 2.50662828) * np.exp(-0.5 * (delta / game_sigma) ** 2)

def x2(delta, game_sigma):
    return delta / game_sigma ** 2

def x3(delta, game_sigma):
    return -delta / game_sigma ** 2

def x4(delta, game_sigma):
    return -(game_sigma ** 2 - delta ** 2) / game_sigma ** 3

class GradientDescent:
    def __init__(self, num_teams, learning_rate=0.01, monthly_decay=0.9, season_reset_mult=0.2):
        self.learning_rate = learning_rate
        self.monthly_decay = monthly_decay
        self.season_reset_mult = season_reset_mult
        self.games = np.empty((0, 5), int)
        self.home_advantage = 5
        self.sigma = 12
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

        mus_adjustments_home = x2(realities_home + away_ratings - self.home_advantage - home_ratings, game_sigmas) * weights
        mus_adjustments_away = x2(realities_away + home_ratings + self.home_advantage - away_ratings, game_sigmas) * weights
        grad_sigma = x4(realities_home - expectations_home, game_sigmas) * weights
        grad_home_advantage = x3(realities_home - expectations_home, game_sigmas) * weights

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

        home_objective = np.sum(x1(realities_home + away_ratings - self.home_advantage - home_ratings, game_sigmas) * weights)
        away_objective = np.sum(x1(realities_away + home_ratings + self.home_advantage - away_ratings, game_sigmas) * weights)
        sigma_objective = np.sum(x1(realities_home - expectations_home, game_sigmas) * weights) * 0.1

        return home_objective + away_objective + sigma_objective

    def add_game(self, timestamp, team_home, team_away, score_home, score_away):
        self.games = np.vstack([self.games, np.array([timestamp, team_home, team_away, score_home, score_away])])
        self.games = self.games[-6000:]
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

        for stat in stats:
            weight = 0.994 ** abs((date - stat['Date']).days)

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

        self.my_team_id = {}
        self.countdown = 3500
        self.past_pred = []
        self.last_season = -1
        self.season_start = 0
        self.lr = None
        self.lr_retrain = 0

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
            self.gd_model.predict(self.my_team_id[home_id], self.my_team_id[away_id]),
            *home_ff,
            *away_ff
        ]

    def place_bets(self, summary: pd.DataFrame, opps: pd.DataFrame, inc: tuple[pd.DataFrame, pd.DataFrame]):
        games_increment, players_increment = inc

        last_fit = None

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

                        pred = self.lr.predict_proba(np.array([input_arr]))[0, 1]

                        pred = sigmoid(inverse_sigmoid(pred) * conf)

                        odds_home = current['OddsH']
                        odds_away = current['OddsA']

                        min_home_odds = (1 / pred - 1) * 1.2 + 1 + 0.00
                        min_away_odds = (1 / (1 - pred) - 1) * 1.2 + 1 + 0.00

                        if odds_home >= min_home_odds:
                            bets.at[i, 'BetH'] = min_bet * 2

                        if odds_away >= min_away_odds:
                            bets.at[i, 'BetA'] = min_bet * 2

        return bets
