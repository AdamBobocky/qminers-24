import json
import plotly.graph_objects as go

with open('src/meta_model/data.json', 'r') as file:
    data = json.load(file)

# Example P&L data (replace this with your actual data)
pnl = 0
pnl_history = [pnl]

for el in data:
    # {'index': '19626', 'neutral': 0, 'playoff': 0, 'date': '1992-04-07 00:00:00', 'season': 17, 'score': 13, 'my_pred': 0.8135907717999672, 'mkt_pred': 0.8204547225222816, 'odds_home': 1.1715005717120743, 'odds_away': 5.353319173866908, 'outcome': 1, 'inputs': [0.0, 0.7272665624401353, 0.5718569415807854, 0.4354914514331333, 0.619008109437756, -1.0, 277.4593024178033, 4.6604413986206055], 'coefs': [[0.3882274467340216], [0.0, -0.06739559754120401, -0.3510749852737827, 0.6120463811105197, -0.06526022102312355, -0.4011082640431251, 0.0011262947435209147, 0.08482271663461109]]}
    if el['my_pred'] * el['odds_home'] > 1.05:
        pnl -= 1
        if el['outcome'] == 1:
            pnl += el['odds_home']
        pnl_history.append(pnl)

    if (1 - el['my_pred']) * el['odds_away'] > 1.05:
        pnl -= 1
        if el['outcome'] == 0:
            pnl += el['odds_away']
        pnl_history.append(pnl)

# Generate an index for the P&L data (e.g., days or transactions)
x = list(range(1, len(pnl_history) + 1))

# Create a Plotly figure
fig = go.Figure()

# Add the P&L line chart
fig.add_trace(go.Scatter(x=x, y=pnl_history, mode='lines+markers', name='P&L'))

# Customize layout
fig.update_layout(
    title='Profit & Loss History',
    xaxis_title='Period (Bets)',
    yaxis_title='Profit & Loss',
    template='plotly_white'
)

# Show the plot
fig.show()
