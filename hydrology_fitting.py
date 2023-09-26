import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
import csv

df = pd.read_csv('raw data/divisions of rainfall runoff.csv')
df = df[["start time", "cumulative rainfall", "rainfall duration", "maximum 48-hour rainfall", "Wan San cumulative rainfall", 'rainfall in the previous 10 days', "rising flow", "flood peak"]]
name = ["cumulative rainfall", "rainfall duration", "maximum 48-hour rainfall", "Wan San cumulative rainfall", 'rainfall in the previous 10 days', "rising flow"]

df.dropna(inplace=True)
X = df.drop(["flood peak", "start time"], axis=1)
y = df["flood peak"]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled)
X_scaled = X_scaled.values
y = y.values

corr_matrix = np.corrcoef(X_scaled, y, rowvar=False)
feature_weights = np.std(X_scaled, 0, ddof=1) * corr_matrix[-1, :-1]
train_data = df.loc[190: 209, :]

with open('process data/multi-source_hydrology_fitting.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerows([])

print('Start testing......')
k = 1
for i in train_data.index:
    x = df[name].loc[i]
    a, b, c, d, e, f = x
    df2 = pd.DataFrame(data=[[a, b, c, d, e, f]], columns=name)
    po = PolynomialFeatures(degree=2, interaction_only=False, include_bias=False)
    df2_poly = po.fit_transform(df2)
    df2_dataframe = pd.DataFrame(df2_poly, )
    list5 = []
    df_all = pd.concat((df[name].iloc[:190, :], df2), axis=0, join='outer')

    normal = preprocessing.MinMaxScaler()
    df_allnormal = normal.fit_transform(df_all)
    df_nn = abs(df_allnormal - df_allnormal[-1])
    df_nnnormal = normal.fit_transform(df_nn)

    for n in range(0, 6):
        df_nnnormal[:, n] = df_nnnormal[:, n] * feature_weights[n]

    df_nnnormal = df_nnnormal[:-1, :]
    df_nnnormal_sum = np.sum(df_nnnormal, axis=1)

    m = 5
    N_small = pd.DataFrame(df_nnnormal_sum).sort_values(by=0, ascending=True).head(int(m))
    N_samll_index = pd.DataFrame({'similarity': df_nnnormal_sum}).sort_values(by='similarity', ascending=True)
    N_samll_index = list(N_samll_index.index)[:int(m)]

    df_sim = pd.DataFrame()
    for s in N_small.index:
        df_sim = pd.concat((df_sim, df.iloc[s:s + 1, :]), axis=0, join='outer')

    df_sim["similarity"] = N_small[0].tolist()
    df_sim["name"] = N_samll_index
    df_sim = df_sim[["start time", "name", "similarity", "cumulative rainfall", "flood peak"]]
    df_sim = df_sim.set_index("start time")
    df_sim['test_name'] = i

    list6 = ["similarity"]
    for z in list6:
        df_sim[z] = df_sim[z].apply(lambda x: round(x, 4))
    list7 = ["cumulative rainfall"]
    for z in list7:
        df_sim[z] = df_sim[z].apply(lambda x: round(x, 2))
    list8 = ["name", "flood peak"]
    for z in list8:
        df_sim[z] = df_sim[z].astype(int)

    if k < 2:
        k += 1
        df_sim.to_csv('process data/multi-source_hydrology_fitting.csv', mode='a', index=False, header=True)
    else:
        df_sim.to_csv('process data/multi-source_hydrology_fitting.csv', mode='a', index=False, header=False)

print('Start saving the results......')
df_sim = pd.read_csv('process data/multi-source_hydrology_fitting.csv')
list9 = list(df_sim['test_name'])
list10 = []
for i in list9:
    x = df.iloc[int(i), 7]
    list10.append(x)
df_sim['test_flood peak'] = list10
df_sim.to_csv('process data/multi-source_hydrology_fitting.csv', index=False)