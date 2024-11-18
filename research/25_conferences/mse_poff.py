import math
import json

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def inverse_sigmoid(x):
    return math.log(x / (1 - x))

with open('src/meta_model/data.json', 'r') as f:
    data = json.load(f)

mse_reg = 0
conf_reg = 0
theo_mse_reg = 0
mse_reg_n = 0
mse_poff = 0
conf_poff = 0
theo_mse_poff = 0
mse_poff_n = 0

home_wins = 0
home_exp = 0
home_n = 0

for point in data:
    # mkt = sigmoid(inverse_sigmoid(point['my_pred']) * 1.35)
    mkt = sigmoid((inverse_sigmoid(point['my_pred']) + 0.2) * 1.1) # POFF
    # mkt = point['my_pred']
    inv_mkt = 1 - mkt

    if point['playoff'] == 1:
        mse_poff += (mkt - point['outcome']) ** 2
        conf_poff += abs(inverse_sigmoid(mkt))
        theo_mse_poff += mkt ** 2 * inv_mkt + inv_mkt ** 2 * mkt
        mse_poff_n += 1
        home_wins += point['outcome']
        home_exp += mkt
        home_n += 1
    else:
        mse_reg += (mkt - point['outcome']) ** 2
        conf_reg += abs(inverse_sigmoid(mkt))
        theo_mse_reg += mkt ** 2 * inv_mkt + inv_mkt ** 2 * mkt
        mse_reg_n += 1

print(mse_reg / mse_reg_n, conf_reg / mse_reg_n, theo_mse_reg / mse_reg_n, mse_reg_n)
print(mse_poff / mse_poff_n, conf_poff / mse_poff_n, theo_mse_poff / mse_poff_n, mse_poff_n)
print(home_wins, home_exp, home_n)
