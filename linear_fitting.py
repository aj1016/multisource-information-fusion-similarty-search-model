import pandas as pd
import numpy as np
import csv
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error


'''Getting the dataset'''
data = open('process data/multi-source_fitted data.csv', 'w', encoding='utf-8', newline='')
csv_writer = csv.writer(data)
csv_writer.writerow(["temporal similarity", 'spatial similarity', 'name', 'test_name', 'hydrological similarity', 'flood peak'])
data.close()

data = pd.read_csv('process data/multi-source_fitted data.csv')
# spatial similarity，temporal similarity
df_1 = pd.read_csv('process data/multi-source_spatiotemporal_fitting.csv')
x1 = df_1.iloc[:, 2]
x2 = df_1.iloc[:, 3]
lis_1 = []
lis_2 = []
for i in range(len(x1)):
    lis_1.append(x1[i])
    lis_2.append(x2[i])
data["spatial similarity"] = lis_1
data["temporal similarity"] = lis_2

# name、test——name、hydrological similarity、flood peak
df_3 = pd.read_csv('process data/multi-source_hydrology_fitting.csv')
x3 = df_3.iloc[:, 0]
x4 = df_3.iloc[:, 4]
x5 = df_3.iloc[:, 1]
x6 = df_3.iloc[:, 3]
lis_3, lis_4, lis_5, lis_6 = [], [], [], []
for i in range(len(x3)):
    lis_3.append(x3[i])
    lis_4.append(x4[i])
    lis_5.append(x5[i])
    lis_6.append(x6[i])
data["name"] = lis_3
data["test_name"] = lis_4
data["hydrological similarity"] = lis_5
data["flood peak"] = lis_6

# Obtaining Absolute Difference Characteristic Columns for Flood Peak Predictions
list7 = []
for i in range(0, len(df_3)):
    diff = round(abs(df_3.iloc[i, 3] - df_3.iloc[i, 5]), 3)
    list7.append(diff)
data['absolute differences'] = list7
data.to_csv('process data/multi-source_fitted data.csv', index=False)

sample_data = pd.read_csv('process data/multi-source_fitted data.csv', header=0)

'''Producing data sets'''
# random_state sets the random seed
random_rows = sample_data.sample(n=50, random_state=120)
X = random_rows[['hydrological similarity', "temporal similarity", 'spatial similarity']]
Y = random_rows[['absolute differences']]

'''The regression of linear'''
linear_regressor = LinearRegression()
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=0)
linear_regressor.fit(x_train, y_train)
w = linear_regressor.coef_
w = np.array(w)
print('weight：', w)
print('intercept：', linear_regressor.intercept_)

column_names = ['hydrological similarity', 'temporal similarity', 'spatial similarity']
df = pd.DataFrame(w, columns=column_names)
df.to_csv('process data/linear_weight data.csv', index=False)

'''Validation using test sets'''
Y_pred = linear_regressor.predict(x_test)

'''Model Evaluation'''
mse = mean_squared_error(y_test, Y_pred)
RR = linear_regressor.score(x_test, y_test)
data = {'index': ['Mean squared error', 'Root mean squared error', 'Mean absolute error', 'RR'],
        'value': [mse, np.sqrt(mse), mean_absolute_error(y_test, Y_pred), RR]}
df = pd.DataFrame(data)
df.to_csv('process data/linear_fitting accuracy.csv', index=False)