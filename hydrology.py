import pandas as pd
import numpy as np
from sklearn import preprocessing
import csv
from sklearn.preprocessing import StandardScaler


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


df = pd.read_csv('raw data/divisions of rainfall runoff.csv')
df = df[["start time", "cumulative rainfall", "rainfall duration", "maximum 48-hour rainfall", "Wan San cumulative rainfall", 'rainfall in the previous 10 days', "rising flow", "flood peak"]]
name = ["cumulative rainfall", "rainfall duration", "maximum 48-hour rainfall", "Wan San cumulative rainfall", 'rainfall in the previous 10 days', "rising flow"]

# Remove missing values
df.dropna(inplace=True)
X = df.drop(["flood peak", "start time"], axis=1)
y = df["flood peak"]

# Normalization of features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled)
X_scaled = X_scaled.values
y = y.values

# Calculate the matrix of characteristic correlation coefficients
corr_matrix = np.corrcoef(X_scaled, y, rowvar=False)
feature_weights = np.std(X_scaled, 0, ddof=1) * corr_matrix[-1, :-1]
print('Feature weights：', feature_weights)

'''Finding set and test set in 8:2 ratio (210:52)'''
data_k = []
data_r = []
data_rmse = []
data_rse = []
print('Start testing......')
for k in range(1, 6):
    print('When K={a}, the prediction accuracy is calculated......'.format(a=k))
    predict = []
    true = []
    session = []
    for i in range(210, 262):
        x = df[name].iloc[i:i + 1, :]
        a, b, c, d, e, f = x.loc[i]
        df3 = pd.DataFrame(data=[[a, b, c, d, e, f]], columns=name)

        '''Calculating similarity'''
        df_all = pd.concat((df[name].iloc[:210, :], df3), axis=0, join='outer').reset_index(drop=True)
        normal = preprocessing.MinMaxScaler()
        df_allnormal = normal.fit_transform(df_all)
        df_nn = abs(df_allnormal - df_allnormal[-1])
        df_nnnormal = normal.fit_transform(df_nn)

        for m in range(0, 6):
            df_nnnormal[:, m] = df_nnnormal[:, m]*feature_weights[m]
        df_nnnormal = df_nnnormal[:-1, :]
        df_nnnormal_sum = np.sum(df_nnnormal, axis=1)
        N_small_index = pd.DataFrame({'similarity': df_nnnormal_sum}).sort_values(by='similarity', ascending=True)
        N_small_index = list(N_small_index.index)[:int(k)]

        df_sim = pd.DataFrame()
        for j in N_small_index:
            df_sim = pd.concat((df_sim, df.iloc[j:j + 1, 7:8]), axis=0, join='outer')

        ranks = range(k)
        decay_factor = 0.5
        weights = exponential_decay_weights(ranks, decay_factor)
        df_sim_value = round(np.sum(df_sim["flood peak"].values * np.array(weights)), 2)
        session.append(i)
        predict.append(df_sim_value)
        true.append(df.loc[i]["flood peak"])

    '''evaluation function'''
    y_true = np.array(tuple(true))
    y_predict = np.array(tuple(predict))
    r, rmse = evaluation(y_true, y_predict)
    data_k.append(k)
    data_r.append(round(r, 3))
    data_rmse.append(round(int(rmse)))

print('Start saving all accuracy evaluation scores......')
data = open('evaluation results/hydrology_evaluation.csv', 'w', encoding='utf-8', newline='')
csv_writer = csv.writer(data)
csv_writer.writerow(['K', "correlation coefficient", "Root mean square error"])
data.close()
data = pd.read_csv('evaluation results/hydrology_evaluation.csv')
data["K"] = data_k
data["correlation coefficient"] = data_r
data["Root mean square error"] = data_rmse
data.to_csv('evaluation results/hydrology_evaluation.csv', index=False)
