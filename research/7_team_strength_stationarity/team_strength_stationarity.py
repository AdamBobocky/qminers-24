import pandas as pd
import numpy as np

file_path = 'data/games.csv'
df = pd.read_csv(file_path)

def inverse_sigmoid(x):
    return np.log(x / (1 - x))

df['Inv_OddsH'] = inverse_sigmoid(1 / df['OddsH'])
df['Inv_OddsA'] = inverse_sigmoid(1 / df['OddsA'])

home_avg_odds = df.groupby(['Season', 'HID'])['Inv_OddsH'].mean().reset_index()
home_avg_odds.columns = ['Season', 'TeamID', 'Avg_Odds']

# Calculate the average odds for away games
away_avg_odds = df.groupby(['Season', 'AID'])['Inv_OddsA'].mean().reset_index()
away_avg_odds.columns = ['Season', 'TeamID', 'Avg_Odds']

# Combine the two results
avg_odds = pd.concat([home_avg_odds, away_avg_odds])

# Group by Season and TeamID to get the final average odds per season for each team
final_avg_odds = avg_odds.groupby(['Season', 'TeamID'])['Avg_Odds'].mean().reset_index()

# Display the result
for season in final_avg_odds['Season'].unique():
    print(f"Season: {season}")
    season_data = final_avg_odds[final_avg_odds['Season'] == season]
    print(season_data[['TeamID', 'Avg_Odds']].to_string(index=False))
    print("\n")
