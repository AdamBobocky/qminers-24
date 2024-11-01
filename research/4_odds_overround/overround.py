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

print(df['overround'])

residuals = df['overround'].to_numpy()

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

print('Average overround:', df['overround'].mean()) # Average overround: 1.0540611058387956
print('Negative overrounds:', (df['overround'] < 1.0).mean()) # Negative overrounds: 0.0
