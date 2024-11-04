from datetime import datetime
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression

def date_to_timestamp(date_str):
  date_obj = datetime.strptime(date_str, "%Y-%m-%d")

  return int(date_obj.timestamp() * 1000)

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

            print(new_objective / games_count, end='\r')
            if new_objective > best_objective + 0.000001:
                best_objective = new_objective
                countdown = 10

        print('')

    def predict(self, team_home, team_away):
        return self._get_average_home_advantage() + self.team_advantages[team_home] + self.team_mus[team_home] - self.team_mus[team_away]

df = pd.read_csv('data/games.csv')

my_team_id = {}
num_teams = 0

for i in df.index:
    current = df.loc[i]

    home_id = current['HID']
    away_id = current['AID']

    if home_id not in my_team_id:
        my_team_id[home_id] = num_teams
        num_teams += 1

    if away_id not in my_team_id:
        my_team_id[away_id] = num_teams
        num_teams += 1

model = GradientDescent(num_teams)

past_pred = []
corr_me = []
corr_mkt = []

skip = 2000

my_mse = 0
mkt_mse = 0
n = 0

for i in df.index:
    skip -= 1

    current = df.loc[i]

    timestamp = date_to_timestamp(current['Date'])
    home_id = current['HID']
    away_id = current['AID']
    home_score = current['HSC']
    away_score = current['ASC']
    home_win = current['H']
    odds_home = current['OddsH']
    odds_away = current['OddsA']
    overround = 1 / odds_home + 1 / odds_away
    mkt_pred = 1 / odds_home / overround

    if skip > 0:
        model.add_game(timestamp, my_team_id[home_id], my_team_id[away_id], home_score, away_score)

        continue

    pred = model.predict(my_team_id[home_id], my_team_id[away_id])

    if len(past_pred) > 500:
        np_array = np.array(past_pred[-2000:])
        lr = LogisticRegression()
        lr.fit(np_array[:, :-1], np_array[:, -1])

        win_prob = lr.predict_proba(np.array([pred]).reshape(1, -1))[0, 1]

        corr_me.append(win_prob)
        corr_mkt.append(mkt_pred)

        my_mse += (win_prob - home_win) ** 2
        mkt_mse += (mkt_pred - home_win) ** 2
        n += 1

        r = np.corrcoef(corr_me, corr_mkt)[0, 1]

        print('my_mse   ', my_mse / n)
        print('mkt_mse  ', mkt_mse / n)
        print('corr r   ', r)
        print('corr r2  ', r ** 2)

    past_pred.append([pred, home_win])

    model.add_game(timestamp, my_team_id[home_id], my_team_id[away_id], home_score, away_score)

    model.fit()

    print('next:', len(past_pred))

print('Done')
