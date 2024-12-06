import json
from sklearn.metrics import r2_score

# Load the data from the JSON files
with open('src/meta_model/outcome_data.json', 'r') as file:
    meta_model_data = json.load(file)

with open('data/odds/odds_map.json', 'r') as file:
    odds_map_data = json.load(file)

# Helper function to convert American odds to decimal odds
def american_to_decimal(american_odds):
    american_odds = float(american_odds)
    if american_odds > 0:
        return (american_odds / 100) + 1
    else:
        return (100 / abs(american_odds)) + 1

my_preds = []
mkt_pred = []
pnl = 0
bets = 0
my_mse = 0
mkt_mse = 0
elo_mse = 0
n = 0

# Match the index and extract relevant odds, then convert
for c in meta_model_data:
    meta_index = c['index']
    if meta_index in odds_map_data and c['playoff'] == 0:
        odds = odds_map_data[meta_index]
        odds_home_score = int(odds[3])
        odds_away_score = int(odds[7])
        home_odds = american_to_decimal(odds[2])
        away_odds = american_to_decimal(odds[6])

        odds_delta_score = odds_home_score - odds_away_score
        my_score = c['score']

        if odds_delta_score != my_score:
            print(odds_delta_score, my_score)

        if(int(c['date'][:4]) > 2008):
            continue

        outcome = c['outcome']
        my_pred = c['my_pred']
        elo_pred = c['mkt_pred']
        odds_pred = 1 / home_odds / (1 / home_odds + 1 / away_odds)

        my_mse += (my_pred - outcome) ** 2
        mkt_mse += (odds_pred - outcome) ** 2
        elo_mse += (elo_pred - outcome) ** 2
        n += 1

        CUTOFF = 1

        if my_pred * home_odds > CUTOFF:
            pnl -= 1
            if outcome == 1:
                pnl += home_odds
            bets += 1

        if (1 - my_pred) * away_odds > CUTOFF:
            pnl -= 1
            if outcome == 0:
                pnl += away_odds
            bets += 1

        my_preds.append(my_pred)
        mkt_pred.append(odds_pred)

print('our mse:', my_mse / n, 'odds mse:', mkt_mse / n, 'virt. bookmaker mse:', elo_mse / n, 'n:', n)
print('pnl per bet:', pnl / bets, 'bets:', bets)

r2 = r2_score(my_preds, mkt_pred)

print(f"The r2 correlation between me and odds: {r2}")
