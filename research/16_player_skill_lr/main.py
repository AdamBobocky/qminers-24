import math
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.preprocessing import StandardScaler

# Load the data
players_df = pd.read_csv('data/players.csv')
games_df = pd.read_csv('data/games.csv')

# Initialize the dictionary to hold players' next game outcomes
win_next_dict = {}

# Prepare lists to hold training data
inputs = []
outputs = []

total = len(players_df)

# Iterate players_df in reverse order
for index, row in players_df[::-1].iterrows():
    matching_game = games_df.loc[row['Game']]
    home_id = matching_game['HID']
    home_win = matching_game['H']

    player_is_home_team = home_id == row['Team']
    player_won = home_win if player_is_home_team else (1 - home_win)

    this_player = row['Player']  # Assuming 'Player' column is there
    # Check if player has already played and won next game
    if this_player in win_next_dict and row['MIN'] > 0:
        # Collect the input features for training
        input_data = [
            (1 if player_is_home_team else 0) / row['MIN'], # Whether player is part of home team
            row['PTS'] / row['MIN'],  # Points
            row['ORB'] / row['MIN'],  # Offensive rebounds
            row['DRB'] / row['MIN'],  # Defensive rebounds
            row['AST'] / row['MIN'],  # Assists
            row['STL'] / row['MIN'],  # Steals
            row['BLK'] / row['MIN'],  # Blocks
            row['FGA'] / row['MIN'],  # Field goal attempts
            row['FTA'] / row['MIN'],  # Free throw attempts
            row['TOV'] / row['MIN'],  # Turnovers
            row['PF'] / row['MIN'],   # Personal fouls
        ]

        if not any(math.isnan(x) for x in input_data):
            inputs.append(input_data)

            # The output is whether the player won the next game
            outputs.append(win_next_dict[this_player])

    if row['MIN'] > 0:
        # Update win_next_dict with whether this player won this game
        win_next_dict[this_player] = player_won
    else:
        if this_player in win_next_dict:
            del win_next_dict[this_player]

    if (index + 1) % (total // 1000) == 0:
      progress = 100 - (index + 1) / total * 100
      print(f"Progress: {progress:.1f}%")

# Convert inputs and outputs to numpy arrays
inputs = pd.DataFrame(inputs)
outputs = pd.Series(outputs)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(inputs[::-1], outputs[::-1], test_size=0.2, shuffle=False)

# Initialize the StandardScaler
scaler = StandardScaler()

# Fit and transform the training data, then transform the test data
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train the logistic regression model
model = LogisticRegression(max_iter=1000)
model.fit(X_train_scaled, y_train)

y_pred = model.predict_proba(X_test_scaled)[:, 1]
y_pred_binary = [1 if p >= 0.5 else 0 for p in y_pred]

print(X_test_scaled)
print(y_pred)

# Print the coefficients of the logistic regression model
print("Coefficients: ", model.coef_)
print("Intercept: ", model.intercept_)

accuracy = accuracy_score(y_test, y_pred_binary)
print(f"Accuracy: {accuracy}")

# For Mean Squared Error (regression)
mse = mean_squared_error(y_pred, y_test)
print(f"Mean Squared Error: {mse}")

# TODO: Group by by team and gameid, export keyed predictions to ensamble with my existing model
# See how accuracies improve when done teamH - teamA
