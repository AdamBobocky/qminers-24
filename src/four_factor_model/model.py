import math
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression

def inverse_sigmoid(x):
    return math.log(x / (1 - x))

class Model:
    def __init__(self):
        self.team_stats_average = {}
        self.opponent_stats_average = {}
        self.input_map = {}
        self.prediction_map = {}
        self.past_pred = []
        self.corr_me = []
        self.corr_mkt = []

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

    def get_team_four_factor(self, season, team_id):
        stats = self.team_stats_average[season][team_id]
        opp_stats = self.opponent_stats_average[season][team_id]

        return [
            (stats['FieldGoalsMade'] + 0.5 * stats['3PFieldGoalsMade']) / stats['FieldGoalAttempts'],
            stats['Turnovers'] / (stats['FieldGoalAttempts'] + 0.44 * stats['FreeThrowAttempts'] + stats['Turnovers']),
            stats['OffensiveRebounds'] / (stats['OffensiveRebounds'] + stats['OpponentsDefensiveRebounds']),
            stats['FreeThrowAttempts'] / stats['FieldGoalAttempts'],
            (opp_stats['FieldGoalsMade'] + 0.5 * opp_stats['3PFieldGoalsMade']) / opp_stats['FieldGoalAttempts'],
            opp_stats['Turnovers'] / (opp_stats['FieldGoalAttempts'] + 0.44 * opp_stats['FreeThrowAttempts'] + opp_stats['Turnovers']),
            opp_stats['OffensiveRebounds'] / (opp_stats['OffensiveRebounds'] + opp_stats['OpponentsDefensiveRebounds']),
            opp_stats['FreeThrowAttempts'] / opp_stats['FieldGoalAttempts']
        ]

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

        for i in games_increment.index:
            current = games_increment.loc[i]

            season = current['Season']
            home_id = current['HID']
            away_id = current['AID']
            home_win = current['H']
            odds_home = current['OddsH']
            odds_away = current['OddsA']
            overround = 1 / odds_home + 1 / odds_away
            mkt_pred = 1 / odds_home / overround

            if season not in self.team_stats_average:
                self.team_stats_average[season] = {}
            if season not in self.opponent_stats_average:
                self.opponent_stats_average[season] = {}
            for team_id in [home_id, away_id]:
                prev_season = season - 1
                if team_id not in self.team_stats_average[season]:
                    if prev_season in self.team_stats_average and team_id in self.team_stats_average[prev_season]:
                        self.team_stats_average[season][team_id] = {
                            'FieldGoalsMade': self.team_stats_average[prev_season][team_id]['FieldGoalsMade'] * 0.2,
                            '3PFieldGoalsMade': self.team_stats_average[prev_season][team_id]['3PFieldGoalsMade'] * 0.2,
                            'FieldGoalAttempts': self.team_stats_average[prev_season][team_id]['FieldGoalAttempts'] * 0.2,
                            'Turnovers': self.team_stats_average[prev_season][team_id]['Turnovers'] * 0.2,
                            'OffensiveRebounds': self.team_stats_average[prev_season][team_id]['OffensiveRebounds'] * 0.2,
                            'OpponentsDefensiveRebounds': self.team_stats_average[prev_season][team_id]['OpponentsDefensiveRebounds'] * 0.2,
                            'FreeThrowAttempts': self.team_stats_average[prev_season][team_id]['FreeThrowAttempts'] * 0.2,
                            'Score': self.team_stats_average[prev_season][team_id]['Score'] * 0.2,
                            'Games': self.team_stats_average[prev_season][team_id]['Games'] * 0.2
                        }
                    else:
                        self.team_stats_average[season][team_id] = {
                            'FieldGoalsMade': 0,
                            '3PFieldGoalsMade': 0,
                            'FieldGoalAttempts': 0,
                            'Turnovers': 0,
                            'OffensiveRebounds': 0,
                            'OpponentsDefensiveRebounds': 0,
                            'FreeThrowAttempts': 0,
                            'Score': 0,
                            'Games': 0
                        }
            for team_id in [home_id, away_id]:
                prev_season = season - 1
                if team_id not in self.opponent_stats_average[season]:
                    if prev_season in self.opponent_stats_average and team_id in self.opponent_stats_average[prev_season]:
                        self.opponent_stats_average[season][team_id] = {
                            'FieldGoalsMade': self.opponent_stats_average[prev_season][team_id]['FieldGoalsMade'] * 0.2,
                            '3PFieldGoalsMade': self.opponent_stats_average[prev_season][team_id]['3PFieldGoalsMade'] * 0.2,
                            'FieldGoalAttempts': self.opponent_stats_average[prev_season][team_id]['FieldGoalAttempts'] * 0.2,
                            'Turnovers': self.opponent_stats_average[prev_season][team_id]['Turnovers'] * 0.2,
                            'OffensiveRebounds': self.opponent_stats_average[prev_season][team_id]['OffensiveRebounds'] * 0.2,
                            'OpponentsDefensiveRebounds': self.opponent_stats_average[prev_season][team_id]['OpponentsDefensiveRebounds'] * 0.2,
                            'FreeThrowAttempts': self.opponent_stats_average[prev_season][team_id]['FreeThrowAttempts'] * 0.2,
                            'Score': self.opponent_stats_average[prev_season][team_id]['Score'] * 0.2,
                            'Games': self.opponent_stats_average[prev_season][team_id]['Games'] * 0.2
                        }
                    else:
                        self.opponent_stats_average[season][team_id] = {
                            'FieldGoalsMade': 0,
                            '3PFieldGoalsMade': 0,
                            'FieldGoalAttempts': 0,
                            'Turnovers': 0,
                            'OffensiveRebounds': 0,
                            'OpponentsDefensiveRebounds': 0,
                            'FreeThrowAttempts': 0,
                            'Score': 0,
                            'Games': 0
                        }

            self.team_stats_average[season][home_id]['FieldGoalsMade'] += current['HFGM']
            self.team_stats_average[season][home_id]['3PFieldGoalsMade'] += current['HFG3M']
            self.team_stats_average[season][home_id]['FieldGoalAttempts'] += current['HFGA']
            self.team_stats_average[season][home_id]['Turnovers'] += current['HTOV']
            self.team_stats_average[season][home_id]['OffensiveRebounds'] += current['HORB']
            self.team_stats_average[season][home_id]['OpponentsDefensiveRebounds'] += current['ADRB']
            self.team_stats_average[season][home_id]['FreeThrowAttempts'] += current['HFTA']
            self.team_stats_average[season][home_id]['Score'] += current['HSC']
            self.team_stats_average[season][home_id]['Games'] += 1

            self.team_stats_average[season][away_id]['FieldGoalsMade'] += current['AFGM']
            self.team_stats_average[season][away_id]['3PFieldGoalsMade'] += current['AFG3M']
            self.team_stats_average[season][away_id]['FieldGoalAttempts'] += current['AFGA']
            self.team_stats_average[season][away_id]['Turnovers'] += current['ATOV']
            self.team_stats_average[season][away_id]['OffensiveRebounds'] += current['AORB']
            self.team_stats_average[season][away_id]['OpponentsDefensiveRebounds'] += current['ADRB']
            self.team_stats_average[season][away_id]['FreeThrowAttempts'] += current['AFTA']
            self.team_stats_average[season][away_id]['Score'] += current['ASC']
            self.team_stats_average[season][away_id]['Games'] += 1

            # Opponent
            self.opponent_stats_average[season][away_id]['FieldGoalsMade'] += current['HFGM']
            self.opponent_stats_average[season][away_id]['3PFieldGoalsMade'] += current['HFG3M']
            self.opponent_stats_average[season][away_id]['FieldGoalAttempts'] += current['HFGA']
            self.opponent_stats_average[season][away_id]['Turnovers'] += current['HTOV']
            self.opponent_stats_average[season][away_id]['OffensiveRebounds'] += current['HORB']
            self.opponent_stats_average[season][away_id]['OpponentsDefensiveRebounds'] += current['ADRB']
            self.opponent_stats_average[season][away_id]['FreeThrowAttempts'] += current['HFTA']
            self.opponent_stats_average[season][away_id]['Score'] += current['HSC']
            self.opponent_stats_average[season][away_id]['Games'] += 1

            self.opponent_stats_average[season][home_id]['FieldGoalsMade'] += current['AFGM']
            self.opponent_stats_average[season][home_id]['3PFieldGoalsMade'] += current['AFG3M']
            self.opponent_stats_average[season][home_id]['FieldGoalAttempts'] += current['AFGA']
            self.opponent_stats_average[season][home_id]['Turnovers'] += current['ATOV']
            self.opponent_stats_average[season][home_id]['OffensiveRebounds'] += current['AORB']
            self.opponent_stats_average[season][home_id]['OpponentsDefensiveRebounds'] += current['ADRB']
            self.opponent_stats_average[season][home_id]['FreeThrowAttempts'] += current['AFTA']
            self.opponent_stats_average[season][home_id]['Score'] += current['ASC']
            self.opponent_stats_average[season][home_id]['Games'] += 1

            if i in self.input_map:
                self.past_pred.append([*self.input_map[i], home_win])

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

            season = current['Season']
            home_id = current['HID']
            away_id = current['AID']
            odds_home = current['OddsH']
            odds_away = current['OddsA']
            overround = 1 / odds_home + 1 / odds_away
            mkt_pred = 1 / odds_home / overround
            mkt_pred_factor = inverse_sigmoid(mkt_pred)

            if current['Date'] == summary.iloc[0]['Date']:
                if season in self.team_stats_average and home_id in self.team_stats_average[season] and away_id in self.team_stats_average[season] and self.team_stats_average[season][home_id]['Games'] >= 6 and self.team_stats_average[season][away_id]['Games'] >= 6:
                    new_frame_inputs = [mkt_pred_factor, *self.get_team_four_factor(season, home_id), *self.get_team_four_factor(season, away_id)]
                    self.input_map[i] = new_frame_inputs
                    if len(self.past_pred) >= 1500:
                        np_data = np.array(self.past_pred)[-2000:]
                        X = np_data[:, :-1]
                        y = np_data[:, -1]
                        lr = LogisticRegression(max_iter=10000)
                        lr.fit(X, y)
                        # print('coefficients:', lr.coef_, 'intercept:', lr.intercept_)
                        pred = lr.predict_proba(np.array([new_frame_inputs]))[0, 1]

                        self.prediction_map[i] = pred

                        min_home_odds = (1 / pred - 1) * 1.05 + 1 + 0.03
                        min_away_odds = (1 / (1 - pred) - 1) * 1.05 + 1 + 0.03

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
