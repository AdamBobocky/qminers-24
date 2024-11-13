import math
import json
import numpy as np
import plotly.graph_objects as go
from datetime import datetime
from sklearn.linear_model import LogisticRegression
from scipy.optimize import minimize
from sklearn.metrics import log_loss

def inverse_sigmoid(x):
    return math.log(x / (1 - x))

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def np_sigmoid(z):
    return 1 / (1 + np.exp(-z))

with open('src/meta_model/data.json', 'r') as file:
    data = json.load(file)

data_len = len(data)

last_season = -1
season_start = 0

week_roi = {
    -1: []
}

raw_data = []

pnl = 0
bets = 0
mse = 0

for el in data:
    if el['season'] != last_season:
        last_season = el['season']
        season_start = datetime.strptime(el['date'], '%Y-%m-%d %H:%M:%S')

    date = datetime.strptime(el['date'], '%Y-%m-%d %H:%M:%S')
    days = (date - season_start).days
    week = days // 7
    odds_home = el['odds_home']
    odds_away = el['odds_away']
    pred = el['my_pred']
    # pred = sigmoid(inverse_sigmoid(el['my_pred']) * (days * 0.0014451108374287569 + 0.883186072970893))
    outcome = el['outcome']

    mse += (pred - outcome) ** 2

    min_home_odds = (1 / pred - 1) * 1.3 + 1
    min_away_odds = (1 / (1 - pred) - 1) * 1.3 + 1 # + 0.03

    if odds_home > min_home_odds:
        pnl += (outcome * odds_home) - 1
        bets += 1

    if odds_away > min_away_odds:
        pnl += ((1 - outcome) * odds_away) - 1
        bets += 1

    raw_data.append([days, inverse_sigmoid(pred), outcome])

    if week not in week_roi:
        week_roi[week] = []

    week_roi[week].append([inverse_sigmoid(pred), outcome])
    week_roi[-1].append([inverse_sigmoid(pred), outcome])

sorted_data = sorted(week_roi.items())

for week, data in sorted_data:
    if len(data) > 50:
        np_data = np.array(data)
        X = np_data[:, :-1]
        y = np_data[:, -1]
        lr = LogisticRegression(fit_intercept=False)
        lr.fit(X, y)
        preds = lr.predict_proba(X)[:, 1]
        print(week, lr.coef_[0][0], len(X))

# Fitting
def seasonally_adjusted_probability(logits, days_since_start, x, y):
    return np_sigmoid(logits * (days_since_start * x + y))

def objective(params, predictions, days_since_start, true_labels):
    x, y = params

    adjusted_probs = seasonally_adjusted_probability(predictions, days_since_start, x, y)

    loss = log_loss(true_labels, adjusted_probs)

    return loss

# Sample usage
# Assuming you have numpy arrays for these
np_array = np.array(raw_data)
days_since_start = np_array[:, 0]
predictions = np_array[:, 1]
true_labels = np_array[:, 2]

# Initial guesses for x and y
initial_params = [0.01, 0.5]

# Minimize the objective function
result = minimize(objective, initial_params, args=(predictions, days_since_start, true_labels), method='L-BFGS-B')

# Extract the optimal x and y
optimal_x, optimal_y = result.x

print('Optimal x:', optimal_x)
print('Optimal y:', optimal_y)
print('mse:', mse / data_len)
print('pnl:', pnl, 'pnl per bet', pnl / bets, 'bets:', bets)
