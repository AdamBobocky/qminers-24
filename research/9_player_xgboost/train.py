import json
import math
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from collections import defaultdict
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

indices = np.load('temp/np_indices.npy')
X = np.load('temp/np_frames_X.npy')
y = np.load('temp/np_frames_y.npy')
game_y = np.load('temp/np_game_y.npy')

print('indices', indices.shape)
print('X', X.shape)
print('y', y.shape)
print('game_y', game_y.shape)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, shuffle=False)

# Initialize the XGBoost regressor
xg_reg = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, learning_rate=0.1)

# Train the model
xg_reg.fit(X_train, y_train)

# Predict on test data
y_pred = xg_reg.predict(X_test)

# Calculate mean squared error for regression
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")

relevant_indices = indices[-len(y_test):]
relevant_outcomes = game_y[-len(y_test):]

src_df = []

for i in range(len(y_test)):
  src_df.append({
    'GameId': relevant_indices[i][0],
    'Home': relevant_indices[i][1],
    'Pred': y_pred[i],
    'Target': y_test[i],
    'Outcome': relevant_outcomes[i]
  })

df = pd.DataFrame(src_df)

grouped = df.groupby('GameId')

MULTIPLIER = 1.4
SIGMOID_MULTIPLIER = 0.175

prediction_map = {}

mse = 0
match_mse = 0
n = 0

for group_name, group_df in grouped:
  home_filter = group_df[group_df['Home'] == 1]
  away_filter = group_df[group_df['Home'] == 0]

  if len(home_filter) + len(away_filter) < 16:
    continue

  home_pred = home_filter['Pred'].mean()
  away_pred = away_filter['Pred'].mean()
  outcome = home_filter['Outcome'].mean()

  pred = home_pred - away_pred
  home_win = outcome > 0

  pred_prob = sigmoid(pred * SIGMOID_MULTIPLIER)

  match_mse += (pred_prob - home_win) ** 2
  mse += (pred * MULTIPLIER - outcome) ** 2
  n += 1

  prediction_map[group_name] = pred_prob

print('match_mse', match_mse / n, n)
print('mse', mse / n, n)

with open('temp/xgboost_prediction_map.json', 'w') as json_file:
    json.dump(prediction_map, json_file)
