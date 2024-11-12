import math
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from collections import defaultdict

def inverse_sigmoid(x):
    return math.log(x / (1 - x))

class Model:
    def __init__(self):
        self.team_stats_average = defaultdict(list)
        self.opponent_stats_average = defaultdict(list)
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

    def get_stats(self, date, stats):
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
            weight = 0.994 ** (date - stat['Date']).days

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

    def get_team_four_factor(self, date, team_id):
        stats = self.get_stats(date, self.team_stats_average[team_id])
        opp_stats = self.get_stats(date, self.opponent_stats_average[team_id])

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

            home_id = current['HID']
            away_id = current['AID']
            home_win = current['H']
            odds_home = current['OddsH']
            odds_away = current['OddsA']
            overround = 1 / odds_home + 1 / odds_away
            mkt_pred = 1 / odds_home / overround

            self.team_stats_average[home_id].append({
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
            self.team_stats_average[away_id].append({
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
            self.opponent_stats_average[away_id].append({
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

            self.opponent_stats_average[home_id].append({
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

            if i in self.input_map:
                self.past_pred.append([*self.input_map[i], home_win])

            if i in self.prediction_map:
                self.metrics['my_mse'] += (self.prediction_map[i] - home_win) ** 2
                self.metrics['mkt_mse'] += (mkt_pred - home_win) ** 2
                self.metrics['n'] += 1

                self.corr_me.append(self.prediction_map[i])
                self.corr_mkt.append(mkt_pred)

        min_bet = summary.iloc[0]['Min_bet']

        bets = pd.DataFrame(data=np.zeros((len(opps), 2)), columns=['BetH', 'BetA'], index=opps.index)

        for i in opps.index:
            current = opps.loc[i]

            date = current['Date']
            home_id = current['HID']
            away_id = current['AID']
            odds_home = current['OddsH']
            odds_away = current['OddsA']
            overround = 1 / odds_home + 1 / odds_away
            # mkt_pred = 1 / odds_home / overround
            # mkt_pred_factor = inverse_sigmoid(mkt_pred)

            if current['Date'] == summary.iloc[0]['Date']:
                if home_id in self.team_stats_average and away_id in self.team_stats_average and len(self.team_stats_average[home_id]) > 5 and len(self.team_stats_average[away_id]) > 5:
                    new_frame_inputs = [*self.get_team_four_factor(date, home_id), *self.get_team_four_factor(date, away_id)]
                    self.input_map[i] = new_frame_inputs
                    if len(self.past_pred) >= 1500:
                        np_data = np.array(self.past_pred)[-2000:]
                        X = np_data[:, :-1]
                        y = np_data[:, -1]
                        lr = LogisticRegression(max_iter=10000)
                        lr.fit(X, y)
                        pred = lr.predict_proba(np.array([new_frame_inputs]))[0, 1]

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
