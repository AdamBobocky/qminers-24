import json
import numpy as np
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression

with open('src/meta_model/data.json', 'r') as file:
    data = json.load(file)

starting_year = int(data[0]['date'][:4])

training_frames_vanilla = []
training_frames_fut = []

def get_training_frame_vanilla(inputs, score):
    return [
        *inputs,
        score
    ]

def get_training_frame_fut(year_diff, inputs, score):
    arr = []

    for x in inputs:
        arr.append(x)
        arr.append(x * year_diff)

    return [year_diff, *arr, score]

vanilla_list = []
exp_dec_list = []
fut_list = []
vanilla_mse = 0
exp_dec_mse = 0
fut_mse = 0
n = 0

for i in range(len(data)):
    year_diff = int(data[i]['date'][:4]) - starting_year

    training_frame_vanilla = get_training_frame_vanilla(data[i]['inputs'], data[i]['score'])
    training_frame_fut = get_training_frame_fut(year_diff, data[i]['inputs'], data[i]['score'])

    # Potentially retrain and predict
    if len(training_frames_vanilla) > 12000:
        arr = np.array(training_frames_vanilla)
        model_vanilla = LinearRegression().fit(arr[:, :-1], arr[:, -1])
        pred_vanilla = model_vanilla.predict([training_frame_vanilla[:-1]])[0]
        sample_weights = np.exp(-0.0003 * np.arange(len(training_frames_vanilla)))
        model_exp_dec = LinearRegression().fit(arr[:, :-1], arr[:, -1], sample_weight=sample_weights[::-1])
        pred_exp_dec = model_exp_dec.predict([training_frame_vanilla[:-1]])[0]

        arr_fut = np.array(training_frames_fut)
        model_fut = LinearRegression().fit(arr_fut[:, :-1], arr_fut[:, -1], sample_weight=sample_weights[::-1])
        pred_fut = model_fut.predict([training_frame_fut[:-1]])[0]

        vanilla_mse += (training_frame_vanilla[-1] - pred_vanilla) ** 2
        exp_dec_mse += (training_frame_vanilla[-1] - pred_exp_dec) ** 2
        fut_mse += (training_frame_fut[-1] - pred_fut) ** 2
        n += 1

        vanilla_list.append(pred_vanilla)
        exp_dec_list.append(pred_exp_dec)
        fut_list.append(pred_fut)

        print('----------')
        print('exp:', exp_dec_mse / n, n)
        print('van:', vanilla_mse / n, n)
        print('fut:', fut_mse / n, n)

    training_frames_vanilla.append(training_frame_vanilla)
    training_frames_fut.append(training_frame_fut)

correlation1, _ = pearsonr(vanilla_list, exp_dec_list)
print(f"vanilla_list, exp_dec_list: {correlation1}")

correlation2, _ = pearsonr(vanilla_list, fut_list)
print(f"vanilla_list, fut_list: {correlation2}")

correlation3, _ = pearsonr(exp_dec_list, fut_list)
print(f"exp_dec_list, fut_list: {correlation3}")
