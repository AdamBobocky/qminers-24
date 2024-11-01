import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV file into a DataFrame
file_path = 'data/games.csv'  # Change this to your file path
df = pd.read_csv(file_path)

df = df[(df['OddsH'] != 0) & (df['OddsA'] != 0)]

# Calculate de-vigged probabilities for both home and away teams
# Convert odds to probabilities (1/odds)
df['probH'] = 1 / df['OddsH']
df['probA'] = 1 / df['OddsA']

print(df['OddsH'].unique())

# Normalize the probabilities to remove the vig
df['overround'] = df['probH'] + df['probA']
df['true_probH'] = df['probH'] / df['overround']
df['true_probA'] = df['probA'] / df['overround']

# Set the rolling window (e.g., 5 or 7 events)
rolling_window = 200

# Calculate the Mean Squared Error (MSE) for the rolling averages against the true outcomes
# Assume 'H' column represents whether the home team won (1 if home won, 0 otherwise)
# Similarly, assume 'A' represents if the away team won (1 if away won, 0 otherwise)

# MSE for Home (when home team won, true probability should be close to 1)
df['mse_H'] = (df['true_probH'] - df['H']) ** 2

print(df.info())

# MSE for Away (when away team won, true probability should be close to 1)
df['mse_A'] = (df['true_probA'] - df['A']) ** 2

# Aggregate MSE (you can average them together)
df['mse_total'] = (df['mse_H'] + df['mse_A']) / 2

df['rolling_mse'] = df['mse_total'].rolling(window=rolling_window).mean()

# Plot the rolling de-vigged odds
plt.figure(figsize=(10, 6))

# Plot the rolling averages for de-vigged probabilities
# plt.plot(df.index, df['rolling_probH'], label='Home Team Rolling True Probability', color='blue')
# plt.plot(df.index, df['rolling_probA'], label='Away Team Rolling True Probability', color='red')

# Plot MSE on the same plot (scaled down for visibility, you can adjust this)
plt.plot(df.index, df['rolling_mse'], label='Mean Squared Error', color='green', linestyle='--')

# Customize the plot
plt.title('Rolling True Probabilities and MSE Over Events')
plt.xlabel('Event Index')
plt.ylabel('Probability')
plt.legend()
plt.grid(True)

# Show the plot
plt.show()
