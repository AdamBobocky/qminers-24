import json
from sklearn.metrics import r2_score

# Load the data from the JSON files
with open('src/meta_model/spread_data.json', 'r') as file:
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
my_mae = 0
mkt_mae = 0
binary_match = 0
n = 0

# Match the index and extract relevant odds, then convert
for c in meta_model_data:
    meta_index = c['index']

    if(int(c['date'][:4]) > 2008):
        continue

    if meta_index in odds_map_data and c['playoff'] == 0:
        odds = odds_map_data[meta_index]

        # if odds[0].lower() == 'pk' or odds[4].lower() == 'pk':
        #     continue
        if odds[1].lower() == 'pk' or odds[5].lower() == 'pk':
            continue

        # spread1 = float(odds[0]) # Open
        # spread2 = float(odds[4]) # Open
        spread1 = float(odds[1]) # Close
        spread2 = float(odds[5]) # Close
        if spread1 > spread2:
            spread = -spread2
        else:
            spread = spread1
        odds_home_score = int(odds[3])
        odds_away_score = int(odds[7])
        home_odds = american_to_decimal(odds[2])
        away_odds = american_to_decimal(odds[6])

        odds_delta_score = odds_home_score - odds_away_score
        my_score = c['score']

        if odds_delta_score != my_score:
            print(odds_delta_score, my_score)

        my_pred = c['my_pred']

        # print(spread, odds_delta_score, my_score, my_pred)

        if (my_pred > 0) == (spread > 0):
            binary_match += 1

        my_mae += abs(my_pred - my_score)
        mkt_mae += abs(spread - my_score)
        n += 1

        CUTOFF = 0

        if my_score - spread != 0:
            if my_pred - spread > CUTOFF:
                pnl -= 1
                if my_score - spread > 0:
                    pnl += 1.95
                bets += 1

            if my_pred - spread < -CUTOFF:
                pnl -= 1
                if my_score - spread < 0:
                    pnl += 1.95
                bets += 1

        my_preds.append(my_pred)
        mkt_pred.append(spread)

print('binary corr:', binary_match / n)
print('our mae:', my_mae / n, 'odds mae:', mkt_mae / n, 'n:', n)
print('pnl per bet:', pnl / bets, 'bets:', bets)

r2 = r2_score(my_preds, mkt_pred)

print(f"The r2 correlation between me and odds: {r2}")
