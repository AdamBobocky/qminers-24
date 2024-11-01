import math
import numpy as np
import pandas as pd

# K_FACTOR = 0.001 # 0.24590543770344414
# K_FACTOR = 0.002 # 0.24343762575708858
# K_FACTOR = 0.003 # 0.24168514781308065
# K_FACTOR = 0.004 # 0.24027800170485644
# K_FACTOR = 0.008 # 0.23600597148891506
# K_FACTOR = 0.009 # 0.23512063701517455
# K_FACTOR = 0.02 # 0.22808628401461287
# K_FACTOR = 0.03 # 0.22438787791787826
# K_FACTOR = 0.04 # 0.2220487813645233
# K_FACTOR = 0.05 # 0.22050392804637897
# K_FACTOR = 0.06 # 0.21945808153867052
# K_FACTOR = 0.07 # 0.2187450392260056
K_FACTOR = 0.1 # 0.21776958536336236

class Model:
    elo_map = {}
    games_data = []
    predictions = []
    metrics = {
        'elo_mse': 0,
        'odds_mse': 0,
        'n': 0
    }

    def place_bets(self, summary: pd.DataFrame, opps: pd.DataFrame, inc: tuple[pd.DataFrame, pd.DataFrame]):
        games_increment, players_increment = inc
        print(summary['Date'])
        # print(opps) # Contains betting opportunities
        # print(games_increment) # New games data
        # print(players_increment) # New players data

        self.games_data.append(games_increment)

        for i in games_increment.index:
            current = games_increment.loc[i]

            home_id = current['HID']
            away_id = current['AID']
            home_win = current['H']

            for current_id in [home_id, away_id]:
                if current_id not in self.elo_map:
                    self.elo_map[current_id] = 0

            prediction = sigmoid(self.elo_map[home_id] - self.elo_map[away_id])

            self.elo_map[home_id] += K_FACTOR * (home_win - prediction)
            self.elo_map[away_id] += K_FACTOR * ((1 - home_win) - (1 - prediction))

        for i in opps.index:
            current = opps.loc[i]

            home_id = current['HID']
            away_id = current['AID']
            home_odds = current['OddsH']
            away_odds = current['OddsA']

            for current_id in [home_id, away_id]:
                if current_id not in self.elo_map:
                    self.elo_map[current_id] = 0

            prediction = sigmoid(self.elo_map[home_id] - self.elo_map[away_id])
            odds_prediction = 1 / home_odds / (1 / home_odds + 1 / away_odds)

            self.predictions.append({
                'index': i,
                'elo_pred': prediction,
                'odds_pred': odds_prediction,
                'home_odds': home_odds,
                'away_odds': away_odds
            })

    def end(self):
        merge = pd.concat(self.games_data)

        backtests = {
            'pnl_0%': 0,
            'bets_0%': 0,
            'vig_0%': 0,
            'pnl_10%': 0,
            'bets_10%': 0,
            'vig_10%': 0,
            'pnl_20%': 0,
            'bets_20%': 0,
            'vig_20%': 0
        }

        for current in self.predictions:
            if current['index'] in merge.index:
                home_win = merge.at[current['index'], 'H']
                away_win = merge.at[current['index'], 'A']
                elo_pred = current['elo_pred']
                odds_pred = current['odds_pred']
                home_odds = current['home_odds']
                away_odds = current['away_odds']
                self.metrics['elo_mse'] += (elo_pred - home_win) ** 2
                self.metrics['odds_mse'] += (odds_pred - home_win) ** 2
                self.metrics['n'] += 1

                if elo_pred * home_odds > 1:
                    backtests['pnl_0%'] -= 1
                    if home_win:
                        backtests['pnl_0%'] += home_odds
                    backtests['vig_0%'] += 1 / home_odds + 1 / away_odds - 1
                    backtests['bets_0%'] += 1
                if (1 - elo_pred) * away_odds > 1:
                    backtests['pnl_0%'] -= 1
                    if away_win:
                        backtests['pnl_0%'] += away_odds
                    backtests['vig_0%'] += 1 / home_odds + 1 / away_odds - 1
                    backtests['bets_0%'] += 1

                if elo_pred * home_odds > 1.1:
                    backtests['pnl_10%'] -= 1
                    if home_win:
                        backtests['pnl_10%'] += home_odds
                    backtests['vig_10%'] += 1 / home_odds + 1 / away_odds - 1
                    backtests['bets_10%'] += 1
                if (1 - elo_pred) * away_odds > 1.1:
                    backtests['pnl_10%'] -= 1
                    if away_win:
                        backtests['pnl_10%'] += away_odds
                    backtests['vig_10%'] += 1 / home_odds + 1 / away_odds - 1
                    backtests['bets_10%'] += 1

                if elo_pred * home_odds > 1.2:
                    backtests['pnl_20%'] -= 1
                    if home_win:
                        backtests['pnl_20%'] += home_odds
                    backtests['vig_20%'] += 1 / home_odds + 1 / away_odds - 1
                    backtests['bets_20%'] += 1
                if (1 - elo_pred) * away_odds > 1.2:
                    backtests['pnl_20%'] -= 1
                    if away_win:
                        backtests['pnl_20%'] += away_odds
                    backtests['vig_20%'] += 1 / home_odds + 1 / away_odds - 1
                    backtests['bets_20%'] += 1

        print('elo_mse:', self.metrics['elo_mse'] / self.metrics['n'])
        print('odds_mse:', self.metrics['odds_mse'] / self.metrics['n'])
        print(backtests)

def sigmoid(x):
    return 1 / (1 + math.exp(-x))
