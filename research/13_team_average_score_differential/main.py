import pandas as pd
import numpy as np

file_path = 'data/games.csv'
df = pd.read_csv(file_path)

def inverse_sigmoid(x):
    return np.log(x / (1 - x))

df['ScoreH'] = df['HSC'] - df['ASC']
df['ScoreA'] = df['ASC'] - df['HSC']

df['ScoreHRatio'] = df['HSC'] / (df['HSC'] + df['ASC'])
df['ScoreARatio'] = df['ASC'] / (df['ASC'] + df['HSC'])

home_avg_score = df.groupby(['Season', 'HID'])['ScoreH'].mean().reset_index()
home_avg_score.columns = ['Season', 'TeamID', 'Avg_Score']

away_avg_score = df.groupby(['Season', 'AID'])['ScoreA'].mean().reset_index()
away_avg_score.columns = ['Season', 'TeamID', 'Avg_Score']

home_avg_score_ratio = df.groupby(['Season', 'HID'])['ScoreHRatio'].mean().reset_index()
home_avg_score_ratio.columns = ['Season', 'TeamID', 'Avg_ScoreRatio']

away_avg_score_ratio = df.groupby(['Season', 'AID'])['ScoreARatio'].mean().reset_index()
away_avg_score_ratio.columns = ['Season', 'TeamID', 'Avg_ScoreRatio']

# Combine the two results
avg_score = pd.concat([home_avg_score, away_avg_score])

# Combine the two results
avg_score_ratio = pd.concat([home_avg_score_ratio, away_avg_score_ratio])

final_avg_score = avg_score.groupby(['Season', 'TeamID'])['Avg_Score'].mean().reset_index()

final_avg_score_ratio = avg_score_ratio.groupby(['Season', 'TeamID'])['Avg_ScoreRatio'].mean().reset_index()

# Display the result
for season in final_avg_score['Season'].unique():
    print(f"Season: {season}")
    season_data = final_avg_score[final_avg_score['Season'] == season]
    print(season_data[['TeamID', 'Avg_Score']].to_string(index=False))
    print("\n")

# Sum the wins per team and season for both home and away games
home_wins = df.groupby(['Season', 'HID'])['H'].sum().reset_index()
home_wins.columns = ['Season', 'TeamID', 'Home_Wins']

away_wins = df.groupby(['Season', 'AID'])['A'].sum().reset_index()
away_wins.columns = ['Season', 'TeamID', 'Away_Wins']

# Calculate total games played per team per season
home_games = df.groupby(['Season', 'HID']).size().reset_index(name='Home_Games')
home_games.columns = ['Season', 'TeamID', 'Home_Games']

away_games = df.groupby(['Season', 'AID']).size().reset_index(name='Away_Games')
away_games.columns = ['Season', 'TeamID', 'Away_Games']

# Merge wins and games data
team_stats = pd.merge(home_wins, away_wins, on=['Season', 'TeamID'], how='outer').fillna(0)
team_stats = pd.merge(team_stats, home_games, on=['Season', 'TeamID'], how='outer').fillna(0)
team_stats = pd.merge(team_stats, away_games, on=['Season', 'TeamID'], how='outer').fillna(0)

# Calculate total wins and games for win rate calculation
team_stats['Total_Wins'] = team_stats['Home_Wins'] + team_stats['Away_Wins']
team_stats['Total_Games'] = team_stats['Home_Games'] + team_stats['Away_Games']
team_stats['Win_Rate'] = team_stats['Total_Wins'] / team_stats['Total_Games']

# Merge win rate with average score differential
combined_data = pd.merge(final_avg_score, team_stats[['Season', 'TeamID', 'Win_Rate']], 
                         on=['Season', 'TeamID'], how='left')

# Calculate correlation between Avg_Score and Win_Rate for each season
correlation_results = {}
for season in combined_data['Season'].unique():
    season_data = combined_data[combined_data['Season'] == season]
    correlation = season_data['Avg_Score'].corr(season_data['Win_Rate'])
    correlation_results[season] = correlation
    print(f"Season {season}: Correlation between Avg Score Differential and Win Rate = {correlation:.4f}")
