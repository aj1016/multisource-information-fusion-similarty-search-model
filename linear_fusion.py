import pandas as pd
import numpy as np
import csv


# Define the accuracy evaluation function
def evaluation(y_true, y_predict):
    r = np.corrcoef(y_true, y_predict)[0, 1]
    rmse = np.sqrt(np.mean((y_true - y_predict) ** 2))
    return r, rmse


# Using Exponential Decay to Obtain Weights for Similarity Ranking
def exponential_decay_weights(similarity_ranks, decay_factor):
    weights = np.exp(-decay_factor * np.array(similarity_ranks))
    weights /= np.sum(weights)
    return weights


''' Obtaining predicted flood peaks after considering multidimensional features '''
w = pd.read_csv("process data/linear_weight data.csv", header=0)
df_1 = pd.read_csv("process data/multi-source_hydrology_fusion.csv", header=0)
df_2 = pd.read_csv("process data/multi-source_spatiotemporal_fusion.csv", header=0)
df_1['hydrological similarity'] = df_1['similarity']
df_1['temporal similarity'] = df_2['dissimilarity_time']
df_1['spatial similarity'] = df_2['dissimilarity_space']
data = df_1.iloc[:, [6, 7, 8]]
data = np.array(data)
w = np.array(w)

# Finding the comprehensive similarity
similar_data = []
for i in range(0, len(data)):
    similar = w[0][0]*data[i][0] + w[0][1]*data[i][1] + w[0][2]*data[i][2]
    similar_data.append(similar)
df_1['fusion similarity'] = similar_data
df_1.to_csv('process data/multi-source_fusion data.csv', index=False)

df = pd.read_csv('raw data/divisions of rainfall runoff.csv')
data = pd.read_csv('process data/multi-source_fusion data.csv')
data_k = []
data_r = []
data_rmse = []
for k in range(1, 6):
    print('When K={}, start training......'.format(k))
    predict = []
    true = []
    for i in range(210, 262):
        data2 = data.loc[data['test_name'] == i]
        sorted_data2 = data2.sort_values(by="fusion similarity")
        selected_data2 = sorted_data2.iloc[:k, 3]

        ranks = range(k)
        decay_factor = 0.5
        weights = exponential_decay_weights(ranks, decay_factor)
        mean_value = round(np.sum(selected_data2.values * np.array(weights)), 2)
        predict.append(mean_value)
        true.append(df.loc[i]['flood peak'])

    '''evaluation function'''
    y_true = np.array(tuple(true))
    y_predict = np.array(tuple(predict))
    r, rmse = evaluation(y_true, y_predict)
    data_k.append(k)
    data_r.append(round(r, 3))
    data_rmse.append(round(int(rmse)))

print('Start saving all accuracy evaluation scores......')

data = open('evaluation results/linear_evaluation.csv', 'w', encoding='utf-8', newline='')
csv_writer = csv.writer(data)
csv_writer.writerow(['K', "correlation coefficient", "Root mean square error"])
data.close()

data = pd.read_csv('evaluation results/linear_evaluation.csv')
data["K"] = data_k
data["correlation coefficient"] = data_r
data["Root mean square error"] = data_rmse
data.to_csv('evaluation results/linear_evaluation.csv', index=False)
