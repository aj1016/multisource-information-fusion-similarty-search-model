# -*- coding:utf-8 -*-
# 一切都是梦，欢迎来到python编译工作室#
# 学 习 者:Sunshine
# 开发日期:2022/8/17
# 显示所有列
import os
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn import preprocessing
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import GridSearchCV
import operator
import functools
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objs as go
import plotly.offline as pyo
import csv
from sklearn.preprocessing import StandardScaler

# 显示所有列
pd.set_option('display.max_columns', None)
# 显示所有行
pd.set_option('display.max_rows', None)
# 设置展示宽度，True就是可以换行显示。设置成False的时候不允许换行
pd.set_option('expand_frame_repr', False)
# 显示最多1500列
# pd.set_option('display.max_columns', 1500)
# 横向最多显示多少个字符
# pd.set_option('display.width', 1000)
# 列长度
# pd.set_option('display.max_colwidth', 1500)
# 是否使用“Unicode-东亚宽度”来计算显示文本宽度，启用此功能可能会影响性能，默认是False；
pd.set_option('display.unicode.ambiguous_as_wide', True)
# 是否使用“Unicode-东亚宽度”来计算显示文本宽度，启用此功能可能会影响性能，默认是False；【常被用来让列名和下面内容对齐】
pd.set_option('display.unicode.east_asian_width', True)


# knn最近邻

# 读取三峡区间流量处理文件
path = os.path.dirname(os.path.realpath(__file__)) + '\\原始数据\\三峡区间流量处理.csv'
# os.path.dirname(os.path.realpath(__file__)),获得你刚才所引用的模块所在的绝对路径，__file__为内置属性。
df_data = pd.read_csv(path, encoding='UTF-8')
# 替换存在缺失值的样本 填充平均值，numeric_only=True由于数据中含有日期列，所以仅数据进行运算
df_data.fillna(df_data.mean(numeric_only=True), inplace=True)

# 读取场次降雨径流划分文件
path = os.path.dirname(os.path.realpath(__file__)) + '\\原始数据\\场次降雨径流划分.csv'
# os.path.dirname(os.path.realpath(__file__)),获得你刚才所引用的模块所在的绝对路径，__file__为内置属性。
df = pd.read_csv(path, encoding='UTF-8')
# 替换存在缺失值的样本 填充平均值，numeric_only=True由于数据中含有日期列，所以仅数据进行运算
df.fillna(df.mean(numeric_only=True), inplace=True)


# 生成特定格式的时间特征（处理df数据）
def strtime_to_datetime(list2):
    list1 = []
    for i in list2:
        # 将list2中数据转化成特定的日期格式
        time1stamp = datetime.strptime(i, "%Y-%m-%d %H:%M:%S")
        list1.append(time1stamp)
    return list1


# 调用定义的strtime_to_datetime函数
df["开始时间"] = strtime_to_datetime(df["开始时间"])
df["结束时间"] = strtime_to_datetime(df["结束时间"])


# 生成特定格式的时间特征（处理df_data数据）
def strtime_to_datetime1(list2):
    list1 = []
    for i in list2:
        time1stamp = datetime.strptime(i, "%Y-%m-%d %H:%M")
        list1.append(time1stamp)
    return list1


# 定义精度评价函数
def evaluation(y_true, y_predict):
    # 相关系数
    r = np.corrcoef(y_true, y_predict)[0, 1]
    # 均方根误差
    rmse = np.sqrt(np.mean((y_true - y_predict) ** 2))
    # 相对平方误差
    rse = np.mean((y_true - y_predict) ** 2) / np.mean(y_true ** 2)
    return r, rmse, rse


# 使用指数衰减法获取相似度排名的权重
def exponential_decay_weights(similarity_ranks, decay_factor):
    """
    参数：
    similarity_ranks: 相似度排名的列表或数组
    decay_factor: 衰减因子，控制权重的衰减速度
    返回值：
    weights: 相似度排名的权重列表
    """
    weights = np.exp(-decay_factor * np.array(similarity_ranks))
    weights /= np.sum(weights)  # 归一化权重，使其总和为1
    return weights


'''计算数据框 df 中每行的 "开始时间" 和 "结束时间" 之间最大的 48 小时降雨量，并将结果存储在 "最大48小时降雨" 这一新列中。
代码首先新建了一个空列 "最大48小时降雨"，并创建了一个空列表 lis_1 用于存储最大降雨量的计算结果。
接着，使用 for 循环遍历数据框中每一行，并将 "开始时间" 和 "结束时间" 之间的时间段对应的 "P" 列（降雨量）提取出来，并存储在一个新的数据框 df_need 中。
如果该时间段内的数据点数不超过 48 个，则将该时间段内的降雨量值求和，并将结果存储在 lis_1 中；
如果数据点数超过 48 个，则遍历所有长度为 48 的子序列，并将每个子序列内的降雨量求和，最终选出这些子序列中的最大值，并将其存储在 lis_1 中。
最后，将 lis_1 中的值赋值给数据框的 "最大48小时降雨" 列，完成了最大 48 小时降雨量的计算和存储。'''

df_data["time"] = strtime_to_datetime1(df_data["time"])
df_data = df_data.set_index("time")
df["最大48小时降雨"] = ""
lis_1 = []
# 最大48小时降雨
for i in range(len(df)):
    start = df["开始时间"][i]
    end = df["结束时间"][i]
    df_need = df_data[start:end]["P"]
    if len(df_need) <= 48:
        lis_1.append(sum(df_need))
    if len(df_need) > 48:
        ret = []
        # 遍历 len(nums) - size + 1 个窗口
        for m in range(0, len(df_need) - 48 + 1):
            ret.append(sum(df_need[m:m + 48]))
        lis_1.append(max(ret))
df["最大48小时降雨"] = lis_1


df = df[["开始时间", "累积雨量", "降雨历时", "最大48小时降雨", "万三累积雨量", '前10天降雨量', "起涨流量", "最大洪峰"]]
name1 = ["累积雨量", "降雨历时", "最大48小时降雨", "万三累积雨量", '前10天降雨量']
name2 = ["累积雨量", "降雨历时", "最大48小时降雨", "万三累积雨量", '前10天降雨量', "起涨流量"]
new_df = df[name2]
new_df.to_csv('原始数据/pearson.csv')

# 特征选择并获取权重
# 去除缺失值
df.dropna(inplace=True)

# Split data into features and target variable,选取特征矩阵和目标变量,values.reshape是将X和y的维度进行调整
X = df.drop(['最大洪峰', '开始时间'], axis=1)
y = df['最大洪峰']

# 对特征进行标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 将 numpy.ndarray 转换为 pandas.DataFrame 对象
X_scaled = pd.DataFrame(X_scaled)

# 将 X_scaled 和 y 转换为 NumPy 数组
X_scaled = X_scaled.values
y = y.values

# 计算特征相关系数矩阵
corr_matrix = np.corrcoef(X_scaled, y, rowvar=False)
# 计算特征权重
feature_weights = np.std(X_scaled, 0, ddof=1) * corr_matrix[-1, :-1]
print(feature_weights)

'''按照8：2比例划查找集和测试集（210：52）'''
data_k = []
data_r = []
data_rmse = []
data_rse = []
print('开始测试......')
for k in range(1, 11):
    print('当K={a}时，计算预测精度......'.format(a=k))
    predict = []
    true = []
    session = []
    for i in range(210, 262):
        x = df[name2].iloc[i:i + 1, :]
        # .loc使用的是标签名称索引，而.iloc是根据标签位置索引
        a, b, c, d, e, f = x.loc[i]
        df3 = pd.DataFrame(data=[[a, b, c, d, e, f]], columns=name2)

        '''计算相似度'''
        # 数据合并与重塑（pd.concat);.iloc：根据标签的所在位置，从0开始计数，先选取行再选取列
        # axis=0表示纵轴，方向从上到下，体现出行的增加或减少，outer在列的方向上进行外连接（即求并集）
        df_all = pd.concat((df[name2].iloc[:210, :], df3), axis=0, join='outer').reset_index(drop=True)

        # 归一化(实例化对象normal)
        normal = preprocessing.MinMaxScaler()
        # Min-Max归一化
        df_allnormal = normal.fit_transform(df_all)
        # 与查找值进行相减，并返回绝对值(abs 函数是 Python 内置函数，主要作用就是计算数字的绝对值)
        df_nn = abs(df_allnormal - df_allnormal[-1])

        # 再对df_nn进行归一化,因为想把最不相似的上限定为6，然后面雨量累积乘以3，最大两日降雨降雨乘以2，最大三日降雨乘以2，
        # 增加权重之后，最不相似上限定为10
        # 先拟合数据，然后转化它将其转化为标准形式
        df_nnnormal = normal.fit_transform(df_nn)

        # 增加权重
        for m in range(0, 6):
            df_nnnormal[:, m] = df_nnnormal[:, m]*feature_weights[m]

        # 去除最后一行
        df_nnnormal = df_nnnormal[:-1, :]
        # print(df_nnnormal)

        # 每一行求和
        df_nnnormal_sum = np.sum(df_nnnormal, axis=1)
        # print(df_nnnormal_sum)
        # # 求最小的f个元素；ascending=False表示降序排序，默认升序
        # N_small = pd.DataFrame(df_nnnormal_sum).sort_values(by=0, ascending=True).head(int(m))
        # 求最小的f个元素对应行号
        N_small_index = pd.DataFrame({'相似度': df_nnnormal_sum}).sort_values(by='相似度', ascending=True)
        # print(N_small_index)
        N_small_index = list(N_small_index.index)[:int(k)]  # N_samll_index是列表格式
        # print(N_small_index)
        # print('最相似的三个场次：', N_small_index)

        df_sim = pd.DataFrame()
        for j in N_small_index:
            # 0表示纵轴，方向从上到下，体现出行的增加或减少，outer在列的方向上进行外连接（即求并集）
            df_sim = pd.concat((df_sim, df.iloc[j:j + 1, 7:8]), axis=0, join='outer')

        ranks = range(k)  # 相似度排名列表
        decay_factor = 0.5  # 衰减因子
        weights = exponential_decay_weights(ranks, decay_factor)
        # print(weights)
        # 指定列的第几行数据与列表对应位置相乘并相加
        df_sim_value = round(np.sum(df_sim['最大洪峰'].values * np.array(weights)), 2)

        # df_sim_value = round(df_sim['最大洪峰'].mean(), 2)
        # print('预测的平均最大洪峰为：', df_sim_value)
        session.append(i)
        predict.append(df_sim_value)
        true.append(df.loc[i]['最大洪峰'])

    '''保存场次-洪峰折线图'''
    if k < 4:
        # 创建文件对象
        folder = 'line_chart/K'
        if not os.path.exists(folder):
            os.makedirs(folder)
        data_line = open('line_chart/K/K_{}.csv'.format(k), 'w', encoding='utf-8', newline='')
        #  基于文件对象构建 csv写入对象
        csv_writer = csv.writer(data_line)
        # 构建列表头
        csv_writer.writerow(['场次', "预测值", "真实值"])
        data_line.close()
        # 填入数据
        data_line = pd.read_csv('line_chart/K/K_{}.csv'.format(k))
        data_line["场次"] = session
        data_line["预测值"] = predict
        data_line["真实值"] = true
        data_line.to_csv('line_chart/K/K_{}.csv'.format(k), index=False)
    '''绘制散点图'''
    # 清除之前的绘图
    plt.clf()
    plt.scatter(true, predict)
    # # 绘制拟合线
    # sns.regplot(x=true, y=predict)
    # 绘制参考线
    plt.plot([0, 30000], [0, 30000], '--')
    # 添加标签和标题
    plt.xlabel('True Values')
    plt.ylabel('Predictions')
    plt.title('K={a},True vs. Predicted'.format(a=k))
    # 指定图片保存路径
    figure_save_path = "figure_K"
    if not os.path.exists(figure_save_path):
        os.makedirs(figure_save_path)  # 如果不存在目录figure_save_path，则创建
    plt.savefig(os.path.join(figure_save_path, 'k={a}.png'.format(a=k)))  # 第一个是指存储路径，第二个是图片名字
    # # 显示图像
    # plt.show()

    '''评价函数'''
    # 将列表转化为元组形式
    y_true = np.array(tuple(true))
    y_predict = np.array(tuple(predict))
    r, rmse, rse = evaluation(y_true, y_predict)
    # noinspection PyUnboundLocalVariable
    data_k.append(k)
    data_r.append(round(r, 3))
    data_rmse.append(round(int(rmse)))
    data_rse.append(round(rse, 3))

# # 定义数据
# data = {'相关系数': round(r, 3), '均方误差': int(mse), '相对平方误差': round(rse, 3)}
# data_with_k = {'{}_{}'.format(key, k): value for key, value in data.items()}
# # 打开文件，使用csv.writer对象写入数据
# with open('评价结果/K_evaluation.csv', 'a',  encoding='utf-8', newline='') as csvfile:
#     fieldnames = ['key', 'value']
#     writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
#     # 写入表头
#     writer.writeheader()
#     # 写入数据
#     for key, value in data_with_k.items():
#         writer.writerow({'key': key, 'value': value})

print('开始保存所有精度评价分数......')
# 创建文件对象
data = open('评价结果/K_evaluation.csv', 'w', encoding='utf-8', newline='')
#  基于文件对象构建 csv写入对象
csv_writer = csv.writer(data)
# 构建列表头
csv_writer.writerow(['K值', "相关系数", "均方根误差", "相对平方误差"])
data.close()
# 填入数据
data = pd.read_csv('评价结果/K_evaluation.csv')
data["K值"] = data_k
data["相关系数"] = data_r
data["均方根误差"] = data_rmse
data["相对平方误差"] = data_rse
data.to_csv('评价结果/K_evaluation.csv', index=False)


