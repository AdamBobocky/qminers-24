import json
import numpy as np
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Load the JSON file
file_path = 'src/meta_model/coef_list.json'  # Replace with your JSON file path
with open(file_path, 'r') as f:
    data = json.load(f)

# Extract coefficients
all_coefs = [item['coefs'][0] for item in data]

# Parameters
N = 8

def extrapolate_with_linear_fit(y, n_fit):
    if len(y) == n_fit:  # Ensure there are enough points for fitting
        x = np.arange(len(y))[-n_fit:].reshape(-1, 1)  # Use last N timesteps
        y_fit = np.array(y[-n_fit:])  # Corresponding coefficient values
        model = LinearRegression().fit(x, y_fit)  # Fit a linear model
        return model.predict([[len(y)]])[0]  # Predict the next value
    else:
        return None

# Plot coefficients and extrapolated values over time
fig = go.Figure()

# Plot original coefficients
for idx in range(len(all_coefs[0])):
    value_tuple = []

    for i in range(len(all_coefs)):
        last_n_coefs = [item[idx] for item in all_coefs[i-N:i]]
        predicted = extrapolate_with_linear_fit(last_n_coefs, N)
        value_tuple.append([all_coefs[i][idx], predicted])

    fig.add_trace(go.Scatter(
        y=[item[0] for item in value_tuple],
        mode='lines+markers',
        name=f'Coefficient {idx}'
    ))

    fig.add_trace(go.Scatter(
        y=[item[1] for item in value_tuple],
        mode='lines',
        line=dict(dash='dot'),  # Dotted line style
        name=f'Extrapolated {idx}'
    ))

fig.update_layout(
    title='Coefficients and Extrapolated Values over Time',
    xaxis_title='Time Step',
    yaxis_title='Coefficient Value'
)

fig.show()
