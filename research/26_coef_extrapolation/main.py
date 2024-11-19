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

# Plot coefficients over time
fig = go.Figure()
for idx in range(len(all_coefs[0])):
    fig.add_trace(go.Scatter(
        y=[item[idx] for item in all_coefs],
        mode='lines+markers',
        name=f'Coefficient {idx}'
    ))
    print(f'Adding {idx}')
fig.update_layout(
    title='Coefficients over Time',
    xaxis_title='Time Step',
    yaxis_title='Coefficient Value'
)
fig.show()

def extrapolate_with_linear_fit(coefs_list, n_fit):
    predicted_coefs = []
    for coef_idx in range(len(coefs_list[0])):  # Iterate over each coefficient
        y = [coefs[coef_idx] for coefs in coefs_list]  # Extract specific coefficient over timesteps
        if len(y) >= n_fit:  # Ensure there are enough points for fitting
            x = np.arange(len(y))[-n_fit:].reshape(-1, 1)  # Use last N timesteps
            y_fit = np.array(y[-n_fit:])  # Corresponding coefficient values
            model = LinearRegression().fit(x, y_fit)  # Fit a linear model
            predicted_coefs.append(model.predict([[len(y)]])[0])  # Predict the next value
        else:
            predicted_coefs.append(None)  # Not enough data for fitting
    return np.array(predicted_coefs)

N = 5

# Calculate MSE using linear extrapolation
errors = []
for i in range(N + 2, len(all_coefs)):
    prev_1 = np.array(all_coefs[i-1])
    # prev_2 = np.array(all_coefs[i-2])
    actual = np.array(all_coefs[i])
    last_n_coefs = all_coefs[i-N:i]
    predicted = extrapolate_with_linear_fit(last_n_coefs, N)
    # predicted = prev_1 + (prev_1 - prev_2)  # Linear extrapolation
    mse_og = mean_squared_error(actual, prev_1)
    mse_extr = mean_squared_error(actual, predicted)
    print(actual, predicted, prev_1)
    errors.append([mse_og, mse_extr])

# Display the MSE values
for i, mse_list in enumerate(errors, start=2):
    mse_og, mse_extr = mse_list
    print(f'Time Step {i} mse_og: {mse_og:.4f}; mse_extr: {mse_extr:.4f}')
