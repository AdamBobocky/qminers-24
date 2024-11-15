import math
import json
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

def inverse_sigmoid(x):
    return math.log(x / (1 - x))

with open('src/meta_model/data.json', 'r') as file:
    data = json.load(file)

# {'index': '19626', 'neutral': 0, 'playoff': 0, 'date': '1992-04-07 00:00:00', 'season': 17, 'score': 13, 'my_pred': 0.8135907717999672, 'mkt_pred': 0.8204547225222816, 'odds_home': 1.1715005717120743, 'odds_away': 5.353319173866908, 'outcome': 1, 'inputs': [0.0, 0.7272665624401353, 0.5718569415807854, 0.4354914514331333, 0.619008109437756, -1.0, 277.4593024178033, 4.6604413986206055], 'coefs': [[0.3882274467340216], [0.0, -0.06739559754120401, -0.3510749852737827, 0.6120463811105197, -0.06526022102312355, -0.4011082640431251, 0.0011262947435209147, 0.08482271663461109]]}

inputs = []
mkt_preds = []

for entry in data:
    inputs.append(entry['inputs'])
    mkt_preds.append(inverse_sigmoid(entry['mkt_pred']))

# Convert inputs and target into a DataFrame for easier handling
X = np.array(inputs)  # Features
y = np.array(mkt_preds)  # Target (mkt_pred)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Initialize the Logistic Regression model
model = LinearRegression()

# Fit the model on the training data
model.fit(X_scaled, y)

# Predict on the test data
y_pred = model.predict(X_scaled)

# Evaluate the model
mse = mean_squared_error(y, y_pred)
r2 = r2_score(y, y_pred)

# Print the results
print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}")

# If you want to check the model's coefficients
np.set_printoptions(precision=6, suppress=True)
print("Model coefficients:", model.coef_)
