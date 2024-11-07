# ROI model

Modular way to input any features and see whether it can find a profit.

The target variables are (binary_win * odds) for home and away.

## XGBRegressor (my_xgboost.py)

This one loses money pretty heavily, sample size not sufficient to get accurate ROI estimate.

## Linear regression (my_lr.py)

This seems to be bringing new information, even if not breaking into profitability for real.
