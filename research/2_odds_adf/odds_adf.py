import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
import statsmodels.api as sm

# Load the CSV file into a DataFrame
file_path = 'data/games.csv'  # Change this to your file path
df = pd.read_csv(file_path)

df = df[(df['OddsH'] != 0) & (df['OddsA'] != 0)]

# Calculate de-vigged probabilities for both home and away teams
# Convert odds to probabilities (1/odds)
df['probH'] = 1 / df['OddsH']
df['probA'] = 1 / df['OddsA']

# Normalize the probabilities to remove the vig
df['overround'] = df['probH'] + df['probA']
df['true_probH'] = df['probH'] / df['overround']
df['true_probA'] = df['probA'] / df['overround']

# Calculate the Mean Squared Error (MSE) for the rolling averages against the true outcomes
# Assume 'H' column represents whether the home team won (1 if home won, 0 otherwise)
# Similarly, assume 'A' represents if the away team won (1 if away won, 0 otherwise)

# MSE for Home (when home team won, true probability should be close to 1)
df['mse_H'] = (df['true_probH'] - df['H']) ** 2

# MSE for Away (when away team won, true probability should be close to 1)
df['mse_A'] = (df['true_probA'] - df['A']) ** 2

# Aggregate MSE (you can average them together)
df['mse_total'] = (df['mse_H'] + df['mse_A']) / 2

residuals = df['mse_total'].to_numpy()

# Step 7: Test for stationarity using Augmented Dickey-Fuller (ADF) test
adf_result = adfuller(residuals)
print('ADF Statistic:', adf_result[0])
print('p-value:', adf_result[1])
print('Critical Values:', adf_result[4])

if adf_result[1] < 0.05:
    print("The residuals are stationary (reject the null hypothesis).")
else:
    print("The residuals are non-stationary (fail to reject the null hypothesis).")

# Step 8: Autocorrelation plot to check for patterns in residuals
sm.graphics.tsa.plot_acf(residuals, lags=40)
plt.show()
