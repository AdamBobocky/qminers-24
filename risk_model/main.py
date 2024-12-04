import json
import math
import random
import numpy as np

with open('risk_model/data.json', 'r') as file:
    data = json.load(file)

def shuffle_data(data):
    random.shuffle(data)

    return data

def kelly_fraction(odds, prob):
    return prob - (1 - prob) / (odds - 1)

def make_bet_vectorized(participants, my_pred, odds, outcome):
    random_free_wins = np.random.uniform(0, 1, len(participants['free_wins']))
    adjusted_outcome = (random_free_wins < participants['free_wins']) | (outcome == 1)

    # Calculate base fractions (Kelly Criterion)
    base_fractions = kelly_fraction(odds, my_pred)
    base_fractions = np.maximum(0, base_fractions)  # Only bet when Kelly fraction is positive

    # Calculate bet amounts
    risk_bankrolls = participants['risk_bankrolls']
    bankrolls = participants['bankrolls']
    risk_level = participants['risk_level']
    intended_reg_amt = bankrolls * base_fractions * risk_level
    intended_risk_amt = risk_bankrolls * base_fractions * 3.0
    bet_amts = np.clip(intended_reg_amt + intended_risk_amt, 5, 100)
    real_reg_bet = intended_reg_amt / bet_amts
    real_risk_bet = intended_risk_amt / bet_amts

    # Only process bets if they are valid (bankroll sufficient)
    valid_bets = (bankrolls - bet_amts) >= 0
    real_reg_bet = real_reg_bet * valid_bets
    real_risk_bet = real_risk_bet * valid_bets

    # Adjust bankrolls based on the outcome
    risk_bankrolls -= real_risk_bet
    risk_bankrolls += real_risk_bet * odds * adjusted_outcome
    bankrolls -= real_reg_bet
    bankrolls += real_reg_bet * odds * adjusted_outcome

    participants['bankrolls'] = bankrolls

def simulate_tournament(c_data, kelly_fraction_multiplier, team_count=16):
    its = 500
    first_place = 0
    second_place = 0
    third_place = 0
    opp_profit = 0

    # Generate opponents kelly multipliers
    # And their edges and their correlations to each other

    for _ in range(its):
        participants = {
            'risk_bankrolls': np.ones(team_count) * 0,
            'bankrolls': np.ones(team_count) * 1000,
            'risk_level': np.random.uniform(0.5, 3.0, team_count),
            'model_quality': np.random.uniform(0.2, 0.5, team_count),
            'noise_level': np.random.uniform(0.01, 0.04, team_count),
            'free_wins': np.random.uniform(0.00, 0.02, team_count)
        }

        participants['risk_level'][0] = kelly_fraction_multiplier
        participants['risk_bankrolls'][0] = 250.0
        participants['bankrolls'][0] = 750.0
        participants['model_quality'][0] = 1.0
        participants['noise_level'][0] = 0.0
        participants['free_wins'][0] = 0.0

        data = shuffle_data(c_data)[:6000]

        for i, bet in enumerate(data):
            my_pred = bet['mkt_pred'] * (1.0 - participants['model_quality']) + bet['my_pred'] * participants['model_quality'] + np.random.uniform(-0.5 * participants['noise_level'], 0.5 * participants['noise_level'])
            odds_home = bet['odds_home']
            odds_away = bet['odds_away']
            outcome = bet['outcome']

            if i == 1000:
                participants['bankrolls'] += participants['risk_bankrolls']
                participants['risk_bankrolls'] = 0.0

            # Home bet
            make_bet_vectorized(participants, my_pred, odds_home, outcome == 1)
            # Away bet
            make_bet_vectorized(participants, 1 - my_pred, odds_away, outcome == 0)

        if np.sort(participants['bankrolls'])[-1] == participants['bankrolls'][0]:
            first_place += 1
        if np.sort(participants['bankrolls'])[-2] == participants['bankrolls'][0]:
            second_place += 1
        if np.sort(participants['bankrolls'])[-3] == participants['bankrolls'][0]:
            third_place += 1

    print('sim:', kelly_fraction_multiplier, first_place / its, second_place / its, third_place / its)

    return first_place / its, second_place / its, third_place / its

def grid_search_optimize(data, bounds=(0.5, 1.2), steps=10):
    best_multiplier = None
    max_log_profit = -np.inf
    for multiplier in np.linspace(bounds[0], bounds[1], steps):
        log_profit = simulate_tournament(data, multiplier)

# Optimize Kelly fraction
optimal_fraction, max_log_profit = grid_search_optimize(data)

print(f'Optimal Kelly Fraction Multiplier: {optimal_fraction}')
print(f'Maximum Log Profit: {max_log_profit}')
