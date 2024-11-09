import json
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt

# Step 1: Load JSON data
with open('src/model2/data.json', 'r') as file:
    data = json.load(file)['pred_list']

# Step 2: Extract features and targets
my_exp = np.array([item['my_exp'] for item in data]).reshape(-1, 1)
outcome = np.array([item['outcome'] for item in data])

# Step 3: Fit Logistic Regression model
logistic_model = LogisticRegression()
logistic_model.fit(my_exp, outcome)
predicted_probs = logistic_model.predict_proba(my_exp)[:, 1]

# Step 4: Calculate Mean Squared Error for Logistic Regression
logistic_mse = mean_squared_error(outcome, predicted_probs)
print("Mean Squared Error for Logistic Regression:", logistic_mse)

# Step 5: Plot Calibration Curve
true_prob, predicted_prob = calibration_curve(outcome, predicted_probs, n_bins=80)

plt.figure(figsize=(8, 6))
plt.plot(predicted_prob, true_prob, marker='o', label="Logistic Regression")
plt.plot([0, 1], [0, 1], linestyle="--", label="Perfectly Calibrated")

plt.xlabel("Predicted Probability")
plt.ylabel("True Probability")
plt.title("Calibration Curve")
plt.legend()
plt.show()
