import pandas as pd
import numpy as np

# Load dataset
df = pd.read_csv('data/2013_vegas.csv')
# df = pd.read_csv('data/2012_vegas_playoff.csv')

# Function to convert moneyline odds to implied probabilities
def moneyline_to_probability(odds):
    if odds > 0:
        return 100 / (odds + 100)
    else:
        return -odds / (-odds + 100)

# Calculate devigged probabilities for each game using Pinnacle_ML odds
# We need to group by GameId, then calculate devigged probabilities for each game
grouped = df.groupby('GameId')
results = []

mse = 0
mse_n = 0

for game_id, group in grouped:
    if len(group) == 2:  # Only proceed if we have two rows per game
        team1 = group.iloc[0]
        team2 = group.iloc[1]

        # Convert Pinnacle_ML to probabilities
        prob_team1 = moneyline_to_probability(team1['Pinnacle_ML'])
        prob_team2 = moneyline_to_probability(team2['Pinnacle_ML'])

        # Normalize probabilities to remove vigorish (vig)
        total_prob = prob_team1 + prob_team2
        devigged_prob_team1 = prob_team1 / total_prob
        devigged_prob_team2 = prob_team2 / total_prob
        # print('ok', prob_team1, prob_team2, total_prob)

        # Get the result for each team (1 if Win, 0 if Loss)
        result_team1 = 1 if team1['Result'] == 'W' else 0
        result_team2 = 1 if team2['Result'] == 'W' else 0

        # Calculate squared errors
        sq_error_team1 = (devigged_prob_team1 - result_team1) ** 2
        sq_error_team2 = (devigged_prob_team2 - result_team2) ** 2

        print(sq_error_team1)

        # if sq_error_team1 != NaN:
        if not np.isnan(devigged_prob_team1):
          mse += sq_error_team1
          mse_n += 1

        # Append squared errors to the results list
        # results.extend([sq_error_team1, sq_error_team2])

# Calculate Mean Squared Error
# mse = np.mean(results)
print("Mean Squared Error (MSE) for devigged probabilities against results:", mse / mse_n)
