# Maintain rolling average of past 10 games player stats and their score difference
# Train logistic regression which takes last 730 days of the data, and fits it

import math
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.preprocessing import StandardScaler

# Load the data
players_df = pd.read_csv('data/players.csv')
games_df = pd.read_csv('data/games.csv')

players_df = players_df[players_df['MIN'] > 3]

merged_df = players_df.merge(games_df, left_on='Game', right_index=True)

merged_df['ScoreDifference'] = merged_df.apply(
    lambda row: row['HSC'] - row['ASC'] if row['Team'] == row['HID'] else row['ASC'] - row['HSC'],
    axis=1
)

merged_df['ScoreDifferencePerMinute'] = merged_df['ScoreDifference'] / merged_df['MIN']
merged_df['PTSPerMinute'] = merged_df['PTS'] / merged_df['MIN']
merged_df['ORBPerMinute'] = merged_df['ORB'] / merged_df['MIN']
merged_df['DRBPerMinute'] = merged_df['DRB'] / merged_df['MIN']
merged_df['ASTPerMinute'] = merged_df['AST'] / merged_df['MIN']
merged_df['STLPerMinute'] = merged_df['STL'] / merged_df['MIN']
merged_df['BLKPerMinute'] = merged_df['BLK'] / merged_df['MIN']
merged_df['FGAPerMinute'] = merged_df['FGA'] / merged_df['MIN']
merged_df['FTAPerMinute'] = merged_df['FTA'] / merged_df['MIN']
merged_df['TOVPerMinute'] = merged_df['TOV'] / merged_df['MIN']
merged_df['PFPerMinute'] = merged_df['PF'] / merged_df['MIN']

merged_df = merged_df.sort_values(by=['Player', 'Game'])

dist_X = merged_df[['ORBPerMinute', 'DRBPerMinute', 'ASTPerMinute', 'STLPerMinute', 'BLKPerMinute', 'FGAPerMinute', 'FTAPerMinute', 'TOVPerMinute', 'PFPerMinute']]
dist_X = dist_X.dropna()

merged_df['Past10AvgScoreDiffPerMinute'] = merged_df.groupby('Player')['ScoreDifferencePerMinute'].transform(lambda x: x.rolling(10, min_periods=10).mean())
merged_df['Past10AvgPTSPerMinute'] = merged_df.groupby('Player')['PTSPerMinute'].transform(lambda x: x.rolling(10, min_periods=10).mean())
merged_df['Past10AvgORBPerMinute'] = merged_df.groupby('Player')['ORBPerMinute'].transform(lambda x: x.rolling(10, min_periods=10).mean())
merged_df['Past10AvgDRBPerMinute'] = merged_df.groupby('Player')['DRBPerMinute'].transform(lambda x: x.rolling(10, min_periods=10).mean())
merged_df['Past10AvgASTPerMinute'] = merged_df.groupby('Player')['ASTPerMinute'].transform(lambda x: x.rolling(10, min_periods=10).mean())
merged_df['Past10AvgSTLPerMinute'] = merged_df.groupby('Player')['STLPerMinute'].transform(lambda x: x.rolling(10, min_periods=10).mean())
merged_df['Past10AvgBLKPerMinute'] = merged_df.groupby('Player')['BLKPerMinute'].transform(lambda x: x.rolling(10, min_periods=10).mean())
merged_df['Past10AvgFGAPerMinute'] = merged_df.groupby('Player')['FGAPerMinute'].transform(lambda x: x.rolling(10, min_periods=10).mean())
merged_df['Past10AvgFTAPerMinute'] = merged_df.groupby('Player')['FTAPerMinute'].transform(lambda x: x.rolling(10, min_periods=10).mean())
merged_df['Past10AvgTOVPerMinute'] = merged_df.groupby('Player')['TOVPerMinute'].transform(lambda x: x.rolling(10, min_periods=10).mean())
merged_df['Past10AvgPFPerMinute'] = merged_df.groupby('Player')['PFPerMinute'].transform(lambda x: x.rolling(10, min_periods=10).mean())

merged_df = merged_df.dropna()

# merged_df = merged_df.dropna(subset=['Past10AvgScoreDiffPerMinute'])
# merged_df = merged_df.dropna(subset=['Past10AvgPTSPerMinute'])
# merged_df = merged_df.dropna(subset=['Past10AvgORBPerMinute'])
# merged_df = merged_df.dropna(subset=['Past10AvgDRBPerMinute'])
# merged_df = merged_df.dropna(subset=['Past10AvgASTPerMinute'])
# merged_df = merged_df.dropna(subset=['Past10AvgSTLPerMinute'])
# merged_df = merged_df.dropna(subset=['Past10AvgBLKPerMinute'])
# merged_df = merged_df.dropna(subset=['Past10AvgFGAPerMinute'])
# merged_df = merged_df.dropna(subset=['Past10AvgFTAPerMinute'])
# merged_df = merged_df.dropna(subset=['Past10AvgTOVPerMinute'])
# merged_df = merged_df.dropna(subset=['Past10AvgPFPerMinute'])

# X = merged_df[['Past10AvgPTSPerMinute', 'Past10AvgORBPerMinute', 'Past10AvgDRBPerMinute', 'Past10AvgASTPerMinute', 'Past10AvgSTLPerMinute', 'Past10AvgBLKPerMinute', 'Past10AvgFGAPerMinute', 'Past10AvgFTAPerMinute', 'Past10AvgTOVPerMinute', 'Past10AvgPFPerMinute']]
X = merged_df[['Past10AvgORBPerMinute', 'Past10AvgDRBPerMinute', 'Past10AvgASTPerMinute', 'Past10AvgSTLPerMinute', 'Past10AvgBLKPerMinute', 'Past10AvgFGAPerMinute', 'Past10AvgFTAPerMinute', 'Past10AvgTOVPerMinute', 'Past10AvgPFPerMinute']]
y = merged_df['Past10AvgScoreDiffPerMinute']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

dist_X = dist_X[-len(y_test):]

model = LinearRegression(fit_intercept=False)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)
dist_X.rename(columns={
    'ORBPerMinute': 'Past10AvgORBPerMinute',
    'DRBPerMinute': 'Past10AvgDRBPerMinute',
    'ASTPerMinute': 'Past10AvgASTPerMinute',
    'STLPerMinute': 'Past10AvgSTLPerMinute',
    'BLKPerMinute': 'Past10AvgBLKPerMinute',
    'FGAPerMinute': 'Past10AvgFGAPerMinute',
    'FTAPerMinute': 'Past10AvgFTAPerMinute',
    'TOVPerMinute': 'Past10AvgTOVPerMinute',
    'PFPerMinute': 'Past10AvgPFPerMinute'
}, inplace=True)
y_dist_pred = model.predict(dist_X)

mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")

print("Model coefficients:")
print(f"Intercept: {model.intercept_}")
print(f"Coefficients: {model.coef_}")

print(y_pred)

# Distribution stuff

import distfit
import matplotlib.pyplot as plt

model = distfit.distfit(distr=['norm']) # Loggamma with loc=-0.401 scale=0.159 fits pretty well

# Fit the distributions to your data
model.fit_transform(y_dist_pred)

print(model.model)

# Visualize the distribution and the fitted models
model.plot()
plt.show()
