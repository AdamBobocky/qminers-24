import random
import math
import numpy as np
import pandas as pd
from scipy.stats import norm

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

K_FACTOR = 0.0001

# Generating synthetic data
np.random.seed(42)

# Constants
NUM_TEAMS = 10
NUM_MATCHES = 3000000
true_team_strengths = np.random.normal(0, 1, NUM_TEAMS)  # True strengths of each team

# Synthetic dataset of matches
matches = []
for _ in range(NUM_MATCHES):
    team_a, team_b = np.random.choice(NUM_TEAMS, size=2, replace=False)
    prob = sigmoid(true_team_strengths[team_a] - true_team_strengths[team_b])
    outcome = 1 if random.random() < prob else 0
    matches.append((team_a, team_b, outcome, prob))

matches_df = pd.DataFrame(matches, columns=['Team_A', 'Team_B', 'Outcome', 'Prob'])

# Bayesian Inference parameters
team_strengths = np.zeros(NUM_TEAMS)  # Initialize estimated team strengths

# Bayesian Inference Update Function
def elo_update(team_a, team_b, outcome):
    """Update team strength based on match result using Bayesian inference."""
    # Team performance difference
    team_a = int(team_a)
    team_b = int(team_b)
    strength_diff = team_strengths[team_a] - team_strengths[team_b]

    # Likelihood of observed outcome based on skill difference
    # Assume a normal likelihood with variance representing match outcome noise
    prob_a_wins = sigmoid(strength_diff)
    prob_b_wins = sigmoid(-strength_diff)

    team_strengths[team_a] += K_FACTOR * (outcome - prob_a_wins)
    team_strengths[team_b] += K_FACTOR * ((1 - outcome) - prob_b_wins)

    return prob_a_wins

# Perform Bayesian inference over matches
for _, match in matches_df.iterrows():
    team_a, team_b, outcome, prob = match['Team_A'], match['Team_B'], match['Outcome'], match['Prob']
    my_pred = elo_update(team_a, team_b, outcome)

    # print(my_pred, prob)

# Final estimated strengths
team_strengths_df = pd.DataFrame({
    'Team': range(NUM_TEAMS),
    'Estimated_Strength': team_strengths + (true_team_strengths.min() - team_strengths.min()),
    'True_Strength': true_team_strengths
})

# Display estimated and true strengths for comparison
print(team_strengths_df.sort_values(by='Estimated_Strength', ascending=False))
