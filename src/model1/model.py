import numpy as np
import pandas as pd
import xgboost as xgb
import math
from sklearn.linear_model import LogisticRegression

K_FACTOR = 0.15
HOME_FACTOR = 0.5

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

class Model:
    def __init__(self):
        self.team_stats_average = {}
        self.player_stats_average = {}
        self.team_players = {}

        self.fourfactor_prediction_map = {}
        self.xgb_prediction_map = {}
        self.prediction_map = {}
        self.corr_me = []
        self.corr_mkt = []
        self.ensamble_training_frames = []
        self.games_memory = {}
        self.indices = []
        self.frames_X = []
        self.frames_y = []
        self.game_y = []
        self.xg_reg = False
        self.lr = False
        self.retrain_countdown = 80000
        self.lr_retrain_countdown = 0

        self.metrics = {
            'my_mse': 0,
            'mkt_mse': 0,
            'n': 0
        }

        self.bet_opps = 0
        self.bet_volume = 0
        self.bet_count = 0
        self.bet_sum_odds = 0

        self.elo_map = {}

    def place_bets(self, summary: pd.DataFrame, opps: pd.DataFrame, inc: tuple[pd.DataFrame, pd.DataFrame]):
        games_increment, players_increment = inc

        if self.bet_count > 0:
            print()
            print('Opps:', self.bet_opps, 'Bets:', self.bet_count, 'Volume:', self.bet_volume, 'Avg odds:', self.bet_sum_odds / self.bet_count)

        self.process_games_increment(games_increment, players_increment)

        self.process_players_increment(players_increment)

        min_bet = summary.iloc[0]["Min_bet"]

        bets = pd.DataFrame(data=np.zeros((len(opps), 2)), columns=["BetH", "BetA"], index=opps.index)

        if self.xg_reg != False:
            for i in opps.index:
                current = opps.loc[i]

                self.bet_opps += 1

                if current['Date'] == summary.iloc[0]['Date']:
                    my_pred = self.predict(i, current['Season'], current['HID'], current['AID'])

                    if my_pred != -1:
                        odds_home = current['OddsH']
                        odds_away = current['OddsA']

                        if my_pred * odds_home > 1.1 and odds_home < 5:
                            bets.at[i, 'BetH'] = min_bet

                            self.bet_volume += min_bet
                            self.bet_count += 1
                            self.bet_sum_odds += odds_home

                        if (1 - my_pred) * odds_away > 1.1 and odds_away < 5:
                            bets.at[i, 'BetA'] = min_bet

                            self.bet_volume += min_bet
                            self.bet_count += 1
                            self.bet_sum_odds += odds_away

        if self.metrics['n'] > 0:
            r = np.corrcoef(self.corr_me, self.corr_mkt)[0, 1]
            r_squared = r ** 2

            print('')
            print('my_mse   ', self.metrics['my_mse'] / self.metrics['n'], self.metrics['n'])
            print('mkt_mse  ', self.metrics['mkt_mse'] / self.metrics['n'], self.metrics['n'])
            print('corr r   ', r)
            print('corr r2  ', r_squared)

        return bets

    def process_games_increment(self, games_increment: pd.DataFrame, players_increment: pd.DataFrame):
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

            self.games_memory[i] = {
                'home_id': current['HID'],
                'home_score': current['HSC'],
                'away_score': current['ASC']
            }

            if season not in self.team_stats_average:
                self.team_stats_average[season] = {}
            for team_id in [home_id, away_id]:
                if team_id not in self.team_stats_average[season]:
                    self.team_stats_average[season][team_id] = {
                        'FieldGoalsMade': 0,
                        '3PFieldGoalsMade': 0,
                        'FieldGoalAttempts': 0,
                        'Turnovers': 0,
                        'OffensiveRebounds': 0,
                        'OpponentsDefensiveRebounds': 0,
                        'FreeThrowAttempts': 0
                    }

            self.team_stats_average[season][home_id]['FieldGoalsMade'] += current['HFGM']
            self.team_stats_average[season][home_id]['3PFieldGoalsMade'] += current['HFG3M']
            self.team_stats_average[season][home_id]['FieldGoalAttempts'] += current['HFGA']
            self.team_stats_average[season][home_id]['Turnovers'] += current['HTOV']
            self.team_stats_average[season][home_id]['OffensiveRebounds'] += current['HORB']
            self.team_stats_average[season][home_id]['OpponentsDefensiveRebounds'] += current['ADRB']
            self.team_stats_average[season][home_id]['FreeThrowAttempts'] += current['HFTA']

            self.team_stats_average[season][away_id]['FieldGoalsMade'] += current['AFGM']
            self.team_stats_average[season][away_id]['3PFieldGoalsMade'] += current['AFG3M']
            self.team_stats_average[season][away_id]['FieldGoalAttempts'] += current['AFGA']
            self.team_stats_average[season][away_id]['Turnovers'] += current['ATOV']
            self.team_stats_average[season][away_id]['OffensiveRebounds'] += current['AORB']
            self.team_stats_average[season][away_id]['OpponentsDefensiveRebounds'] += current['ADRB']
            self.team_stats_average[season][away_id]['FreeThrowAttempts'] += current['AFTA']

            home_players = players_increment[(players_increment['Game'] == i) & (players_increment['Team'] == current['HID'])]
            away_players = players_increment[(players_increment['Game'] == i) & (players_increment['Team'] == current['AID'])]

            for current_id in [home_id, away_id]:
                if current_id not in self.elo_map:
                    self.elo_map[current_id] = 0

            elo_factor = self.elo_map[home_id] - self.elo_map[away_id] + HOME_FACTOR
            elo_prediction = sigmoid(elo_factor)

            if i in self.xgb_prediction_map:
                xgb_factor = self.xgb_prediction_map[i]
                four_factor = self.get_team_four_factor(season, home_id) - self.get_team_four_factor(season, away_id)
                self.ensamble_training_frames.append([xgb_factor, elo_factor, four_factor, home_win])

            if i in self.prediction_map:
                self.metrics['my_mse'] += (self.prediction_map[i] - home_win) ** 2
                self.metrics['mkt_mse'] += (mkt_pred - home_win) ** 2
                self.metrics['n'] += 1

                self.corr_me.append(self.prediction_map[i])
                self.corr_mkt.append(mkt_pred)

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
                non_zero_df = df[df['MIN'] > 0]
                np_array = non_zero_df[['MIN', 'FGM', 'FGA', 'FG3M', 'FG3A', 'FTM', 'FTA', 'ORB', 'DRB', 'RB', 'AST', 'STL', 'BLK', 'TOV', 'PF', 'PTS']].to_numpy()
                raw_inputs = np_array[:, 1:] / np_array[:, 0].reshape(-1, 1)
                inputs = np.mean(raw_inputs, axis=0)

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

    def fit_ensamble(self):
        np_data = np.array(self.ensamble_training_frames)[-2000:]
        X = np_data[:, :3]
        y = np_data[:, 3]

        self.lr = LogisticRegression()
        self.lr.fit(X, y)

        print('')
        print('coefficients:', self.lr.coef_)
        print('coefficients:', self.lr.coef_)
        print('coefficients:', self.lr.coef_)

    def fit_xgboost(self):
        self.xg_reg = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, learning_rate=0.1)

        self.xg_reg.fit(np.array(self.frames_X)[-100000:], np.array(self.frames_y)[-100000:])

    def get_team_four_factor(self, season, team_id):
        if season not in self.team_stats_average:
            return 0

        if team_id not in self.team_stats_average[season]:
            return 0

        stats = self.team_stats_average[season][team_id]

        four_factor = 0

        # Effective Field Goal Percentage=(Field Goals Made) + 0.5*3P Field Goals Made))/(Field Goal Attempts)
        four_factor += (stats['FieldGoalsMade'] + 0.5 * stats['3PFieldGoalsMade']) / stats['FieldGoalAttempts'] * 0.4
        # Turnover Rate=Turnovers/(Field Goal Attempts + 0.44*Free Throw Attempts + Turnovers)
        four_factor += stats['Turnovers'] / (stats['FieldGoalAttempts'] + 0.44 * stats['FreeThrowAttempts'] + stats['Turnovers']) * 0.25
        # Offensive Rebounding Percentage = (Offensive Rebounds)/[(Offensive Rebounds)+(Opponent’s Defensive Rebounds)]
        four_factor += stats['OffensiveRebounds'] / (stats['OffensiveRebounds'] + stats['OpponentsDefensiveRebounds']) * 0.2
        # Free Throw Rate=(Free Throws Made)/(Field Goals Attempted) or Free Throws Attempted/Field Goals Attempted
        four_factor += stats['FreeThrowAttempts'] / stats['FieldGoalAttempts'] * 0.15

        return four_factor

    def predict(self, game_id, season, home_id, away_id):
        home_predictions = []
        away_predictions = []

        if home_id not in self.team_players:
            return -1

        if away_id not in self.team_players:
            return -1

        for player_id in self.team_players[home_id]:
            if season in self.player_stats_average and player_id in self.player_stats_average[season] and len(self.player_stats_average[season][player_id]) >= 24:
                df = pd.DataFrame(self.player_stats_average[season][player_id])
                non_zero_df = df[df['MIN'] > 0]
                np_array = non_zero_df[['MIN', 'FGM', 'FGA', 'FG3M', 'FG3A', 'FTM', 'FTA', 'ORB', 'DRB', 'RB', 'AST', 'STL', 'BLK', 'TOV', 'PF', 'PTS']].to_numpy()
                raw_inputs = np_array[:, 1:] / np_array[:, 0].reshape(-1, 1)
                inputs = np.mean(raw_inputs, axis=0)
                home_predictions.append(self.xg_reg.predict([inputs])[0])

        for player_id in self.team_players[away_id]:
            if season in self.player_stats_average and player_id in self.player_stats_average[season] and len(self.player_stats_average[season][player_id]) >= 24:
                df = pd.DataFrame(self.player_stats_average[season][player_id])
                non_zero_df = df[df['MIN'] > 0]
                np_array = non_zero_df[['MIN', 'FGM', 'FGA', 'FG3M', 'FG3A', 'FTM', 'FTA', 'ORB', 'DRB', 'RB', 'AST', 'STL', 'BLK', 'TOV', 'PF', 'PTS']].to_numpy()
                raw_inputs = np_array[:, 1:] / np_array[:, 0].reshape(-1, 1)
                inputs = np.mean(raw_inputs, axis=0)
                away_predictions.append(self.xg_reg.predict([inputs])[0])

        if len(home_predictions) + len(away_predictions) < 16:
            return -1

        four_factor = self.get_team_four_factor(season, home_id) - self.get_team_four_factor(season, away_id)

        self.fourfactor_prediction_map[game_id] = four_factor

        xgb_factor = sum(home_predictions) / len(home_predictions) - sum(away_predictions) / len(away_predictions)

        self.xgb_prediction_map[game_id] = xgb_factor

        elo_factor = self.elo_map[home_id] - self.elo_map[away_id] + HOME_FACTOR

        if len(self.ensamble_training_frames) < 500:
            return -1

        if self.lr_retrain_countdown == 0:
            self.lr_retrain_countdown = 100

            self.fit_ensamble()

        self.lr_retrain_countdown -= 1

        ensamble_pred = self.lr.predict_proba(np.array([xgb_factor, elo_factor, four_factor]).reshape(1, -1))[0, 1]

        self.prediction_map[game_id] = ensamble_pred

        return ensamble_pred
