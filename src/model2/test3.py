import math
import json
import numpy as np
import plotly.graph_objects as go
from datetime import datetime
from sklearn.linear_model import LogisticRegression

def inverse_sigmoid(x):
    return math.log(x / (1 - x))

with open('src/model2/data.json', 'r') as file:
    data = json.load(file)

last_season = -1
season_start = 0

week_roi = {
    -1: []
}

for el in data:
    if el['season'] != last_season:
        last_season = el['season']
        season_start = datetime.strptime(el['date'], '%Y-%m-%d %H:%M:%S')

    date = datetime.strptime(el['date'], '%Y-%m-%d %H:%M:%S')
    week = (date - season_start).days // 7
    pred = el['my_pred']
    outcome = el['outcome']

    if week not in week_roi:
        week_roi[week] = []

    week_roi[week].append([inverse_sigmoid(pred), outcome])
    week_roi[-1].append([inverse_sigmoid(pred), outcome])

sorted_data = sorted(week_roi.items())

for week, data in sorted_data:
    if len(data) > 50:
        np_data = np.array(data)
        X = np_data[:, :-1]
        y = np_data[:, -1]
        lr = LogisticRegression(fit_intercept=False)
        lr.fit(X, y)
        preds = lr.predict_proba(X)[:, 1]
        print(week, lr.coef_[0][0], len(X))
