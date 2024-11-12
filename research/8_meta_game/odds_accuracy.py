# Track inverse sigmoid of implied odds of each team during season, see how accurately it predicts
# outcomes into the future.

import math
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error

file_path = 'data/games.csv'
df = pd.read_csv(file_path)

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def inverse_sigmoid(x):
    return np.log(x / (1 - x))

season_skill_map = {}

HOME_FACTOR = 0.55

predictions = []
pnl = 0
cum_odds = 0
overround_paid = 0
bets = 0

for i in df.index:
    current = df.loc[i]

    neutral = current['N']
    season = current['Season']
    home_id = current['HID']
    away_id = current['AID']
    home_odds = current['OddsH']
    away_odds = current['OddsA']
    home_win = current['H']

    if season not in season_skill_map:
        season_skill_map[season] = {}
    if home_id not in season_skill_map[season]:
        season_skill_map[season][home_id] = []
    if away_id not in season_skill_map[season]:
        season_skill_map[season][away_id] = []

    overround = 1 / home_odds + 1 / away_odds
    home_exp = inverse_sigmoid(1 / home_odds / overround)
    away_exp = inverse_sigmoid(1 / away_odds / overround)

    # Make inferences
    if len(season_skill_map[season][home_id]) >= 24 and len(season_skill_map[season][away_id]) >= 24:
        avg_skill_home = sum(season_skill_map[season][home_id]) / len(season_skill_map[season][home_id])
        avg_skill_away = sum(season_skill_map[season][away_id]) / len(season_skill_map[season][away_id])
        my_pred = sigmoid(avg_skill_home - avg_skill_away + (1 - neutral) * HOME_FACTOR)
        odds_pred = 1 / home_odds / overround

        predictions.append([my_pred, odds_pred, home_win])

        if my_pred * home_odds > 1.2:
            pnl -= 1
            cum_odds += home_odds
            if home_win:
                pnl += home_odds
            overround_paid += overround - 1
            bets += 1
        if (1 - my_pred) * away_odds > 1.2:
            pnl -= 1
            cum_odds += away_odds
            if not home_win:
                pnl += away_odds
            overround_paid += overround - 1
            bets += 1

    season_skill_map[season][home_id].append(home_exp - HOME_FACTOR / 2)
    season_skill_map[season][away_id].append(away_exp + HOME_FACTOR / 2)

print(len(predictions))

# Convert to numpy array for easier manipulation
predictions_array = np.array(predictions)

# Extract my predictions, market predictions, and outcomes
my_predictions = predictions_array[:, 0]
market_predictions = predictions_array[:, 1]
outcomes = predictions_array[:, 2]

my_confidence = np.absolute(my_predictions - 0.5).mean()
market_confidence = np.absolute(market_predictions - 0.5).mean()

my_theo_mse = ((my_predictions ** 2) * (1 - my_predictions) + ((1 - my_predictions) ** 2) * my_predictions).mean()
market_theo_mse = ((market_predictions ** 2) * (1 - market_predictions) + ((1 - market_predictions) ** 2) * market_predictions).mean()

print('my_confidence', my_confidence)
print('market_confidence', market_confidence)
print('my_theo_mse', my_theo_mse)
print('market_theo_mse', market_theo_mse)

# Calculate mean squared error
mse_my = mean_squared_error(outcomes, my_predictions)
mse_market = mean_squared_error(outcomes, market_predictions)

me_market_corr = pearsonr(my_predictions, market_predictions)[0]

# Calculate differences for histogram
differences = my_predictions - market_predictions

# Create a histogram of differences
fig = go.Figure()
fig.add_trace(go.Histogram(x=differences, nbinsx=10, name='Differences'))
fig.update_layout(title='Histogram of Differences Between My and Market Predictions',
                  xaxis_title='Difference (My - Market)',
                  yaxis_title='Count')
# fig.show()

# Output results
print('mse_my', mse_my)
print('mse_market', mse_market)
print('me_market_corr', me_market_corr)
print('avg_odds', cum_odds / bets)
print('avg_pnl', pnl / bets, 'avg_overround', overround_paid / bets, 'bets', bets)
