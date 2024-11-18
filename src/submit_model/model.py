import warnings

warnings.filterwarnings('ignore')

from datetime import datetime

class Exhaustion:
    def __init__(self):
        self.team_rest = {}

    def pre_add_game(self, current, current_players):
        pass

    def add_game(self, current, current_players):
        date = current['Date']
        home_id = current['HID']
        away_id = current['AID']

        self.team_rest[home_id] = date
        self.team_rest[away_id] = date

    def get_input_data(self, home_id, away_id, season, date):
        if home_id not in self.team_rest or away_id not in self.team_rest:
            return None

        home_days = (date - self.team_rest[home_id]).days
        away_days = (date - self.team_rest[away_id]).days

        factor = 0.0

        if home_days <= 1:
            factor += 1.0
        if away_days <= 1:
            factor -= 1.0

        return [
            factor
        ]
import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.functional import softplus
from datetime import datetime
from sklearn.linear_model import LogisticRegression
from collections import defaultdict

class GradientDescent:
    def __init__(self, num_teams=30, monthly_decay=0.8):
        self.num_teams = num_teams
        self.monthly_decay = monthly_decay

        self.team_mus = nn.Parameter(torch.zeros(num_teams))
        self.team_sigmas = torch.ones(num_teams) * 40
        self.home_advantage = nn.Parameter(torch.tensor(5.0))
        self.sigma = nn.Parameter(torch.tensor(12.0))

        self.games = torch.zeros((50000, 5), dtype=torch.long)
        self.game_count = 0

        self.my_team_id = {}
        self.last_season = -1
        self.fit_date = None

    def _check_season(self, season):
        if self.last_season != season:
            self.last_season = season

            self.team_sigmas = torch.ones_like(self.team_sigmas) * 20

    def _get_time_weights(self):
        timestamps = self.games[:self.game_count, 0].to(torch.float32)
        last_ts = timestamps[-1]

        return (self.monthly_decay ** (torch.abs(timestamps - last_ts) / (30 * 24 * 60 * 60 * 1000))).to(timestamps.device)

    def forward(self, weights, idx_start):
        games = self.games[idx_start:self.game_count]
        home_ratings = self.team_mus[games[:, 1]]
        away_ratings = self.team_mus[games[:, 2]]

        expectations_home = self.home_advantage + home_ratings - away_ratings
        realities_home = games[:, 3] - games[:, 4]

        game_sigmas = torch.sqrt(
            self.team_sigmas[games[:, 1]] ** 2 +
            self.team_sigmas[games[:, 2]] ** 2 +
            self.sigma ** 2
        )

        log_value = (-0.5 * ((realities_home - expectations_home) / game_sigmas) ** 2 - torch.log(game_sigmas) - 0.5 * torch.log(torch.tensor(2 * np.pi)))
        return torch.sum(log_value * weights[idx_start:])

    def _fit(self, max_epochs=30):
        weights = self._get_time_weights()
        idx_start = torch.nonzero(weights > 0.004, as_tuple=True)[0][0].item()
        optimizer = optim.Adam([self.team_mus, self.sigma, self.home_advantage], lr=0.04)

        for epoch in range(max_epochs):
            optimizer.zero_grad()
            loss = -self.forward(weights, idx_start)
            loss.backward()
            optimizer.step()

    def add_game(self, current, current_players):
        self._check_season(current['Season'])

        timestamp = int(current['Date'].timestamp() * 1000)
        team_home = self._map_team_id(current['HID'])
        team_away = self._map_team_id(current['AID'])
        score_home = current['HSC']
        score_away = current['ASC']

        self.games[self.game_count] = torch.tensor(
            [timestamp, team_home, team_away, score_home, score_away], dtype=torch.long
        )
        self.game_count += 1

        game_sigma2 = math.sqrt(
            self.team_sigmas[team_home] ** 2 +
            self.team_sigmas[team_away] ** 2 +
            self.sigma ** 2
        )

        self.team_sigmas[team_home] = 1 / torch.sqrt(
            1 / self.team_sigmas[team_home] ** 2 + 1 / game_sigma2)
        self.team_sigmas[team_away] = 1 / torch.sqrt(
            1 / self.team_sigmas[team_away] ** 2 + 1 / game_sigma2)

        self.fit_date = None

    def _map_team_id(self, team_id):
        if team_id not in self.my_team_id:
            self.my_team_id[team_id] = len(self.my_team_id)

        return self.my_team_id[team_id]

    def pre_add_game(self, current, current_players):
        pass

    def get_input_data(self, home_id, away_id, season, date):
        self._check_season(season)

        if self.game_count < 2000:
            return None

        if self.fit_date is None or self.fit_date != date:
            self.fit_date = date
            self._fit()

        team_home = self._map_team_id(home_id)
        team_away = self._map_team_id(away_id)

        game_exp = self.home_advantage + self.team_mus[team_home] - self.team_mus[team_away]
        game_sigma = torch.sqrt(
            self.team_sigmas[team_home] ** 2 +
            self.team_sigmas[team_away] ** 2 +
            self.sigma ** 2
        )

        return [game_exp.item() / game_sigma.item()]
import math
import numpy as np
from collections import defaultdict
from sklearn.linear_model import LogisticRegression

def inverse_sigmoid(x):
    return math.log(x / (1 - x))

class FourFactor:
    def __init__(self):
        # Hyperparameters
        self.lr_required_n = 2000
        # End

        self.team_stats_average = defaultdict(list)
        self.opponent_stats_average = defaultdict(list)

        self.past_pred = []
        self.lr = None
        self.lr_retrain = 0

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

    def pre_add_game(self, current, current_players):
        pass

    def add_game(self, current, current_players):
        val1 = self._get_team_input_data(current['HID'], current['Date'])
        val2 = self._get_team_input_data(current['AID'], current['Date'])

        if val1 is not None:
            self.past_pred.append([*val1, current['H']])
            self.lr_retrain -= 1

        if val2 is not None:
            self.past_pred.append([*val2, current['A']])
            self.lr_retrain -= 1

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

    def _get_team_input_data(self, team_id, date):
        if len(self.team_stats_average[team_id]) <= 5:
            return None

        stats = self._get_stats(date, self.team_stats_average[team_id][-100:])
        opp_stats = self._get_stats(date, self.opponent_stats_average[team_id][-100:])

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

    def get_input_data(self, home_id, away_id, season, date):
        val1 = self._get_team_input_data(home_id, date)
        val2 = self._get_team_input_data(away_id, date)

        if val1 is None or val2 is None:
            return None

        if len(self.past_pred) < self.lr_required_n:
            return None

        if self.lr_retrain <= 0:
            self.lr_retrain = 200

            np_array = np.array(self.past_pred)
            sample_weights = np.exp(-0.0003 * np.arange(len(self.past_pred)))
            self.lr = LogisticRegression(max_iter=10000)
            self.lr.fit(np_array[:, :-1], np_array[:, -1], sample_weight=sample_weights[::-1])

        return [
            inverse_sigmoid(self.lr.predict_proba(np.array([val1]))[0, 1]) -
            inverse_sigmoid(self.lr.predict_proba(np.array([val2]))[0, 1])
        ]
import math
from collections import defaultdict

class NateSilverElo:
    def __init__(self):
        self.elo_map = defaultdict(float)
        self.last_season = -1

    def _new_season(self):
        for key in self.elo_map:
            self.elo_map[key] *= 0.75

    def _win_probability(self, x):
        return 1 / (1 + (math.exp(-x / 175)))

    def pre_add_game(self, current, current_players):
        pass

    def add_game(self, current, current_players):
        season = current['Season']
        home_id = current['HID']
        away_id = current['AID']
        home_score = current['HSC']
        away_score = current['ASC']

        if season > self.last_season:
            self.last_season = season
            self._new_season()

        home_prediction = self._win_probability(self.elo_map[home_id] + 100 - self.elo_map[away_id])
        away_prediction = 1 - home_prediction

        k_factor = self.get_k_factor(home_score - away_score, self.elo_map[home_id] + 100, self.elo_map[away_id])

        self.elo_map[home_id] += k_factor * (current['H'] - home_prediction)
        self.elo_map[away_id] += k_factor * (current['A'] - away_prediction)

    def get_input_data(self, home_id, away_id, season, date):
        if season > self.last_season:
            self.last_season = season
            self._new_season()

        return [
            self.elo_map[home_id] - self.elo_map[away_id] + 100
        ]

    def get_team_strength(self, team_id, is_home, season):
        if season > self.last_season:
            self.last_season = season
            self._new_season()

        return self.elo_map[team_id] + 100 * (0.5 if is_home else -0.5)

    def get_k_factor(self, score_difference, elo_home, elo_away):
        if score_difference > 0:
            return 20 * (score_difference + 3) ** 0.8 / (7.5 + 0.006 * (elo_home - elo_away))
        else:
            return 20 * (-score_difference + 3) ** 0.8 / (7.5 + 0.006 * (elo_away - elo_home))
from collections import defaultdict
from datetime import datetime

class Pythagorean:
    def __init__(self, power=16.5, regularization=120, daily_decay=0.992):
        self.power = power
        self.regularization = regularization
        self.daily_decay = daily_decay

        self.team_map = defaultdict(list)

    def pre_add_game(self, current, current_players):
        pass

    def add_game(self, current, current_players):
        date = current['Date']
        home_id = current['HID']
        away_id = current['AID']
        home_score = current['HSC']
        away_score = current['ASC']

        if type(date) == 'str':
            date = datetime.strptime(date, '%Y-%m-%d')

        home_difference = abs(home_score - away_score) ** self.power * (1 if home_score > away_score else -1)
        away_difference = -home_difference

        self.team_map[home_id].append([date, home_score, away_score])
        self.team_map[away_id].append([date, away_score, home_score])

    def _get_weighted(self, team_id, idx, date):
        return sum([x[idx] * (self.daily_decay ** abs((date - x[0]).days)) for x in self.team_map[team_id][-100:]])

    def get_input_data(self, home_id, away_id, season, date):
        if type(date) == 'str':
            date = datetime.strptime(date, '%Y-%m-%d')

        home_scored = self.regularization + self._get_weighted(home_id, 1, date)
        home_allowed = self.regularization + self._get_weighted(home_id, 2, date)
        away_scored = self.regularization + self._get_weighted(away_id, 1, date)
        away_allowed = self.regularization + self._get_weighted(away_id, 2, date)

        return [
            (home_scored ** self.power) / (home_scored ** self.power + home_allowed ** self.power),
            (away_scored ** self.power) / (away_scored ** self.power + away_allowed ** self.power)
        ]

import math
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

class Model:
    def __init__(self):
        # Hyperparameters
        self.ensamble_required_n = 2000
        nate_silver_elo = NateSilverElo()
        self.model_list = [
            Pythagorean(),
            FourFactor(),
            GradientDescent(),
            Exhaustion(),
            nate_silver_elo
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

    def _game_increment(self, idx, current, current_players):
        season = current['Season']
        date = current['Date']
        home_id = current['HID']
        away_id = current['AID']
        home_win = current['H']
        year = int(str(current['Date'])[0:4])

        if year >= 1994:
            input_arr = self._get_input_features(home_id, away_id, season, date)

            if input_arr is not None:
                self.past_pred.append([*input_arr, home_win])
                self.ensamble_retrain -= 1

        self._handle_metrics(idx, current)

        # if year >= 1986:
        # Let the models create training frames before new data arrives
        for model in self.model_list:
            model.pre_add_game(current, current_players)

        # Add new data to the models
        for model in self.model_list:
            model.add_game(current, current_players)

    def place_bets(self, summary: pd.DataFrame, opps: pd.DataFrame, inc: tuple[pd.DataFrame, pd.DataFrame]):
        games_increment, players_increment = inc

        done = 0
        total = len(games_increment)

        for idx in games_increment.index:
            current = games_increment.loc[idx]
            current_players = players_increment[(players_increment['Game'] == idx) & (players_increment['MIN'] >= 3)]

            self._game_increment(idx, current, current_players)
            done += 1

        min_bet = summary.iloc[0]['Min_bet']
        max_bet = summary.iloc[0]['Max_bet']
        my_bet = max(min_bet, min(max_bet * 0.3, summary.iloc[0]['Bankroll'] * 0.02))

        bets = pd.DataFrame(data=np.zeros((len(opps), 2)), columns=['BetH', 'BetA'], index=opps.index)

        for i in opps.index:
            current = opps.loc[i]

            season = current['Season']
            date = current['Date']
            home_id = current['HID']
            away_id = current['AID']
            playoff = current['POFF'] == 1

            if len(self.past_pred) >= self.ensamble_required_n:
                input_arr = self._get_input_features(home_id, away_id, season, date)

                if input_arr is not None:
                    if self.ensamble_retrain <= 0:
                        self.ensamble_retrain = 300
                        np_array = np.array(self.past_pred)
                        sample_weights = np.exp(-0.0003 * np.arange(len(self.past_pred)))
                        self.ensamble = LogisticRegression(max_iter=10000)
                        self.ensamble.fit(np_array[:, :-1], np_array[:, -1], sample_weight=sample_weights[::-1])

                    self.bet_metrics['opps'] += 1

                    pred = self.ensamble.predict_proba(np.array([input_arr]))[0, 1]

                    self.prediction_map[i] = pred
                    self.input_map[i] = input_arr
                    self.coef_map[i] = [self.ensamble.intercept_.tolist(), *self.ensamble.coef_.tolist()]

                    # Adjust for playoffs
                    adj_pred = pred

                    odds_home = current['OddsH']
                    odds_away = current['OddsA']

                    min_home_odds = (1 / adj_pred - 1) * 1.06 + 1 + 0.03
                    min_away_odds = (1 / (1 - adj_pred) - 1) * 1.06 + 1 + 0.03

                    if odds_home >= min_home_odds:
                        bets.at[i, 'BetH'] = my_bet

                        self.bet_metrics['exp_pnl'] += adj_pred * odds_home - 1
                        self.bet_metrics['volume'] += my_bet
                        self.bet_metrics['count'] += 1
                        self.bet_metrics['sum_odds'] += odds_home

                    if odds_away >= min_away_odds:
                        bets.at[i, 'BetA'] = my_bet

                        self.bet_metrics['exp_pnl'] += (1 - adj_pred) * odds_away - 1
                        self.bet_metrics['volume'] += my_bet
                        self.bet_metrics['count'] += 1
                        self.bet_metrics['sum_odds'] += odds_away

        return bets
