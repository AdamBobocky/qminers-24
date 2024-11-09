import json
from datetime import datetime
import plotly.graph_objects as go

with open('src/model2/data.json', 'r') as file:
    data = json.load(file)

last_season = -1
season_start = 0

season_roi = {}
week_roi = {}

for el in data:
    if el['season'] != last_season:
        last_season = el['season']
        season_start = datetime.strptime(el['date'], '%Y-%m-%d %H:%M:%S')

    # Make a bet
    date = datetime.strptime(el['date'], '%Y-%m-%d %H:%M:%S')
    season = el['season']
    pred = el['my_pred']
    odds_home = el['odds_home']
    odds_away = el['odds_away']
    outcome = el['outcome']
    week = (date - season_start).days // 7

    min_home_odds = (1 / pred - 1) * 1.3 + 1 + 0.04
    min_away_odds = (1 / (1 - pred) - 1) * 1.3 + 1 + 0.04

    if season not in season_roi:
        season_roi[season] = [0, 0, 0]

    if week not in week_roi:
        week_roi[week] = [0, 0, 0]

    season_roi[season][2] += 1
    week_roi[week][2] += 1

    if odds_home > min_home_odds:
        season_roi[season][0] += (outcome * odds_home) - 1
        season_roi[season][1] += 1

        week_roi[week][0] += (outcome * odds_home) - 1
        week_roi[week][1] += 1

    if odds_away > min_away_odds:
        season_roi[season][0] += ((1 - outcome) * odds_away) - 1
        season_roi[season][1] += 1

        week_roi[week][0] += ((1 - outcome) * odds_away) - 1
        week_roi[week][1] += 1

print(season_roi)
print(week_roi)

sorted_data = dict(sorted(week_roi.items()))

x = list(sorted_data.keys())
y1 = [value[0] / max(1, value[1]) + 1 for value in sorted_data.values()]
y2 = [value[1] / value[2] for value in sorted_data.values()]
y3 = [value[2] for value in sorted_data.values()]

fig = go.Figure()

fig.add_trace(go.Scatter(x=x, y=y1, mode='lines+markers', name='ROI', line=dict(color='blue')))
fig.add_trace(go.Scatter(x=x, y=y2, mode='lines+markers', name='Events bet fraction', line=dict(color='red'), yaxis='y2'))
fig.add_trace(go.Scatter(x=x, y=y3, mode='lines+markers', name='Events total', line=dict(color='grey'), yaxis='y3'))

fig.update_layout(
    title="Plot of ROI and bet count",
    xaxis=dict(title="Week"),
    yaxis=dict(title="ROI", titlefont=dict(color="blue"), tickfont=dict(color="blue")),
    yaxis2=dict(title="Events bet fraction", titlefont=dict(color="red"), tickfont=dict(color="red"),
                anchor="x", overlaying="y", side="right"),
    yaxis3=dict(title="Events total", titlefont=dict(color="grey"), tickfont=dict(color="grey"),
                anchor="x", overlaying="y", side="right")
)

# Show the plot
fig.show()
