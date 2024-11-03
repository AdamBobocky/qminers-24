import numpy as np
import pandas as pd
import xgboost as xgb
import math
from sklearn.linear_model import LogisticRegression

# Betting start: 1979-02-07

K_FACTOR = 0.15
HOME_FACTOR = 0.5

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

class Model:
    def __init__(self):
        self.player_stats_average = {}
        self.team_players = {}

        self.prediction_map = {}
        self.predictions = []
        self.games_memory = {}
        self.indices = []
        self.frames_X = []
        self.frames_y = []
        self.game_y = []
        self.xg_reg = False
        self.retrain_countdown = 40000

        self.bet_volume = 0
        self.bet_count = 0
        self.bet_sum_odds = 0

        self.elo_map = {}

    def place_bets(self, summary: pd.DataFrame, opps: pd.DataFrame, inc: tuple[pd.DataFrame, pd.DataFrame]):
        games_increment, players_increment = inc

        if self.bet_count > 0:
            print()
            print('Bets:', self.bet_count, 'Volume:', self.bet_volume, 'Avg odds:', self.bet_sum_odds / self.bet_count)

        self.process_games_increment(games_increment, players_increment)

        self.process_players_increment(players_increment)

        min_bet = summary.iloc[0]["Min_bet"]

        bets = pd.DataFrame(data=np.zeros((len(opps), 2)), columns=["BetH", "BetA"], index=opps.index)

        if self.xg_reg != False:
            for i in opps.index:
                current = opps.loc[i]
                if current['Date'] == summary.iloc[0]['Date']:
                    my_pred = self.predict(i, current['Season'], current['HID'], current['AID'], current['N'])

                    if my_pred != -1:
                        odds_home = current['OddsH']
                        odds_away = current['OddsA']

                        if my_pred * odds_home > 1.15 and odds_home < 3.5:
                            bets.at[i, 'BetH'] = min_bet

                            self.bet_volume += min_bet
                            self.bet_count += 1
                            self.bet_sum_odds += odds_home

                        if (1 - my_pred) * odds_away > 1.15 and odds_away < 3.5:
                            bets.at[i, 'BetA'] = min_bet

                            self.bet_volume += min_bet
                            self.bet_count += 1
                            self.bet_sum_odds += odds_away

        return bets

    def process_games_increment(self, games_increment: pd.DataFrame, players_increment: pd.DataFrame):
        for i in games_increment.index:
            current = games_increment.loc[i]

            home_id = current['HID']
            away_id = current['AID']
            home_win = current['H']
            neutral = current['N']
            odds_home = current['OddsH']
            odds_away = current['OddsA']
            overround = 1 / odds_home + 1 / odds_away
            odds_pred = 1 / odds_home / overround

            self.games_memory[i] = {
                'home_id': current['HID'],
                'home_score': current['HSC'],
                'away_score': current['ASC']
            }

            home_players = players_increment[(players_increment['Game'] == i) & (players_increment['Team'] == current['HID'])]
            away_players = players_increment[(players_increment['Game'] == i) & (players_increment['Team'] == current['AID'])]

            for current_id in [home_id, away_id]:
                if current_id not in self.elo_map:
                    self.elo_map[current_id] = 0

            elo_prediction = sigmoid(self.elo_map[home_id] - self.elo_map[away_id] + (1 - neutral) * HOME_FACTOR)

            if i in self.prediction_map:
                # self.predictions.append([self.prediction_map[i], current['HSC'] - current['ASC']])
                self.predictions.append([self.prediction_map[i], elo_prediction, neutral, home_win, odds_pred])

            self.team_players[current['HID']] = home_players['Player'].tolist()
            self.team_players[current['AID']] = away_players['Player'].tolist()

            self.elo_map[home_id] += K_FACTOR * (home_win - elo_prediction)
            self.elo_map[away_id] += K_FACTOR * ((1 - home_win) - (1 - elo_prediction))

    def process_players_increment(self, players_increment: pd.DataFrame):
        for i in players_increment.index:
            current = players_increment.loc[i]

            game_id = current['Game']
            season = current['Season']
            team_id = current['Team']
            player_id = current['Player']

            matching_game = self.games_memory[game_id]
            home_id = matching_game['home_id']
            home_score = matching_game['home_score']
            away_score = matching_game['away_score']

            if season not in self.player_stats_average:
                self.player_stats_average[season] = {}
            if player_id not in self.player_stats_average[season]:
                self.player_stats_average[season][player_id] = []

            player_is_home_team = home_id == team_id

            player_score_delta = (home_score - away_score) if player_is_home_team else (away_score - home_score)
            player_team_score = home_score if player_is_home_team else away_score

            if len(self.player_stats_average[season][player_id]) >= 24:
                df = pd.DataFrame(self.player_stats_average[season][player_id])
                inputs = df[['MIN', 'FGM', 'FGA', 'FG3M', 'FG3A', 'FTM', 'FTA', 'ORB', 'DRB', 'RB', 'AST', 'STL', 'BLK', 'TOV', 'PF', 'PTS']].mean().to_numpy()
                target = current['PTS']

                self.indices.append([game_id, player_is_home_team])
                self.frames_X.append(inputs)
                self.frames_y.append(target)
                self.game_y.append(player_score_delta)

                self.retrain_countdown -= 1

                if self.retrain_countdown == 0:
                    self.fit_xgboost()

                    self.retrain_countdown = 10000

            # Add this game stats to the player
            self.player_stats_average[season][player_id].append(current)

    def fit_xgboost(self):
        self.xg_reg = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, learning_rate=0.1)

        self.xg_reg.fit(np.array(self.frames_X), np.array(self.frames_y))

    def predict(self, id, season, hid, aid, n):
        home_predictions = []
        away_predictions = []

        for player_id in self.team_players[hid]:
            if season in self.player_stats_average and player_id in self.player_stats_average[season] and len(self.player_stats_average[season][player_id]) >= 24:
                df = pd.DataFrame(self.player_stats_average[season][player_id])
                inputs = df[['MIN', 'FGM', 'FGA', 'FG3M', 'FG3A', 'FTM', 'FTA', 'ORB', 'DRB', 'RB', 'AST', 'STL', 'BLK', 'TOV', 'PF', 'PTS']].mean().to_numpy()
                home_predictions.append(self.xg_reg.predict([inputs])[0])

        for player_id in self.team_players[aid]:
            if season in self.player_stats_average and player_id in self.player_stats_average[season] and len(self.player_stats_average[season][player_id]) >= 24:
                df = pd.DataFrame(self.player_stats_average[season][player_id])
                inputs = df[['MIN', 'FGM', 'FGA', 'FG3M', 'FG3A', 'FTM', 'FTA', 'ORB', 'DRB', 'RB', 'AST', 'STL', 'BLK', 'TOV', 'PF', 'PTS']].mean().to_numpy()
                away_predictions.append(self.xg_reg.predict([inputs])[0])

        if len(home_predictions) + len(away_predictions) < 16:
            return -1

        xbg_factor = sum(home_predictions) / len(home_predictions) - sum(away_predictions) / len(away_predictions)

        self.prediction_map[id] = xbg_factor

        if len(self.predictions) < 200:
            return -1

        np_data = np.array(self.predictions)
        X = np_data[:, :3]
        y = np_data[:, 3]
        odds_y = np_data[:, 4]

        model = LogisticRegression()
        model.fit(X, y)
        metric_preds = model.predict_proba(X)[:, 1]

        r = np.corrcoef(metric_preds, odds_y)[0, 1]
        r_squared = r ** 2

        print('')
        print('my_mse:      ', np.mean((metric_preds - y) ** 2))
        print('odds_mse:    ', np.mean((odds_y - y) ** 2))
        print('r:', r)
        print('r2:', r_squared)

        elo_prediction = sigmoid(self.elo_map[hid] - self.elo_map[aid] + (1 - n) * HOME_FACTOR)

        return model.predict_proba(np.array([xbg_factor, elo_prediction, n]).reshape(1, -1))[0, 1]
