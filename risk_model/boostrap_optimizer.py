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

def make_bet(bankroll, kelly_fraction_multiplier, my_pred, odds, won):
    base_fraction = kelly_fraction(odds, my_pred)
    # bet_amt = bankroll * base_fraction * kelly_fraction_multiplier
    bet_amt = max(5, min(100, bankroll * base_fraction * kelly_fraction_multiplier))

    if base_fraction > 0 and bankroll - bet_amt >= 0:
        # Place a bet
        bankroll -= bet_amt
        if won:
            bankroll += bet_amt * odds

    return bankroll

def simulate_tournament(c_data, kelly_fraction_multiplier):
    its = 4000
    cum_log_wealth = 0
    # kelly_fraction_multiplier = 0.7

    for _ in range(its):
        bankroll = 1000

        data = shuffle_data(c_data)[:6000]

        for bet in data:
            bankroll = make_bet(bankroll, kelly_fraction_multiplier, bet['my_pred'], bet['odds_home'], bet['outcome'] == 1)
            bankroll = make_bet(bankroll, kelly_fraction_multiplier, 1 - bet['my_pred'], bet['odds_away'], bet['outcome'] == 0)

        cum_log_wealth += math.log(bankroll)

    print('sim:', kelly_fraction_multiplier, cum_log_wealth / its)

    return cum_log_wealth / its

def grid_search_optimize(data, bounds=(0.4, 1.2), steps=20):
    best_multiplier = None
    max_log_profit = -np.inf
    for multiplier in np.linspace(bounds[0], bounds[1], steps):
        log_profit = simulate_tournament(data, multiplier)
        if log_profit > max_log_profit:
            max_log_profit = log_profit
            best_multiplier = multiplier
    return best_multiplier, max_log_profit

# Optimize Kelly fraction
optimal_fraction, max_log_profit = grid_search_optimize(data)

print(f'Optimal Kelly Fraction Multiplier: {optimal_fraction}')
print(f'Maximum Log Profit: {max_log_profit}')
