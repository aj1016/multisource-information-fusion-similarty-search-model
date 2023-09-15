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
#显示所有列
pd.set_option('display.max_columns', None)
#显示所有行
pd.set_option('display.max_rows', None)
#设置展示宽度
pd.set_option('expand_frame_repr', False)#True就是可以换行显示。设置成False的时候不允许换行
# pd.set_option('display.max_columns', 1500)#显示最多1500列
# pd.set_option('display.width', 1000)#横向最多显示多少个字符
# pd.set_option('display.max_colwidth', 1500)#列长度
pd.set_option('display.unicode.ambiguous_as_wide', True)# 是否使用“Unicode-东亚宽度”来计算显示文本宽度，启用此功能可能会影响性能，默认是False；
pd.set_option('display.unicode.east_asian_width', True)# 是否使用“Unicode-东亚宽度”来计算显示文本宽度，启用此功能可能会影响性能，默认是False；【常被用来列名和下面内容对齐】


#knn最近邻
#读取三峡区间流量处理文件
path = os.path.dirname(os.path.realpath(__file__)) + '\\原始数据\\三峡区间流量处理.csv'
#os.path.dirname(os.path.realpath(__file__)),获得你刚才所引用的模块所在的绝对路径，__file__为内置属性。
df_data = pd.read_csv(path, encoding='UTF-8')
df_data.fillna(df_data.mean(numeric_only=True), inplace=True)# 替换存在缺失值的样本 填充平均值，numeric_only=True由于数据中含有日期列，所以仅数据进行运算

#读取场次降雨径流划分文件
path = os.path.dirname(os.path.realpath(__file__)) + '\\原始数据\\场次降雨径流划分.csv'
#os.path.dirname(os.path.realpath(__file__)),获得你刚才所引用的模块所在的绝对路径，__file__为内置属性。
df = pd.read_csv(path, encoding='UTF-8')
df.fillna(df.mean(numeric_only=True), inplace=True)#替换存在缺失值的样本 填充平均值，numeric_only=True由于数据中含有日期列，所以仅数据进行运算


def strtime_to_datetime(list2):
    list1 = []
    for i in list2:
        time1stamp = datetime.strptime(i, "%Y-%m-%d %H:%M:%S")
        list1.append(time1stamp)
    return list1
df["开始时间"] = strtime_to_datetime(df["开始时间"])
df["结束时间"] = strtime_to_datetime(df["结束时间"])
def strtime_to_datetime1(list2):
    list1 = []
    for i in list2:
        time1stamp = datetime.strptime(i, "%Y-%m-%d %H:%M")
        list1.append(time1stamp)
    return list1
df_data["time"] = strtime_to_datetime1(df_data["time"])
df_data = df_data.set_index("time")



df["最大48小时降雨"]=""
lis_1 = []
#最大48小时降雨
for i in range(len(df)):
    start= df["开始时间"][i]
    end = df["结束时间"][i]
    df_need=df_data[start:end]["P"]
    if len(df_need) <=48:
        lis_1.append(sum(df_need))
    if len(df_need) > 48:
        ret = []
        # 遍历 len(nums) - size + 1 个窗口
        for m in range(0, len(df_need) - 48+ 1):
            ret.append(sum(df_need[m:m + 48]))
        lis_1.append(max(ret))

df["最大48小时降雨"]=lis_1


df=df[["开始时间","累积雨量","降雨历时","最大48小时降雨","万三累积雨量",'前10天降雨量',"起涨流量","最大洪峰"]]
name1=["累积雨量","降雨历时","最大48小时降雨","万三累积雨量",'前10天降雨量']
name3=["累积雨量","降雨历时","最大48小时降雨","万三累积雨量",'前10天降雨量',"起涨流量"]

n=0
while 1:
    n=n+1
    print("第%s次查找!!!" % n)
#输入

    # print("输入提示：需输入“累积雨量、降雨历时、最大48小时降雨、万三累积雨量、前10天降雨量、起涨流量”六个数值，以逗号隔开")
    # a,b,c,d,e,f=map(float,input("请输入6个数值，以逗号隔开:").split(","))


    print("输入提示:请输入248-261号样本中任意一个样本号")
    label = int(input("请输入:"))
    x = df[name3].iloc[label:label + 1, :]
    a, b, c, d, e, f = x.loc[label]#.loc使用的是标签名称索引，而.iloc是根据标签位置索引
    print('输入样本数据如下:')
    print(df.iloc[label:label + 1, :])
    print('---------------------------------------------------------------------------')

    df2=pd.DataFrame(data=[[a,b,c,d,e]],
                columns = name1,
                index=[0])
    df3 = pd.DataFrame(data=[[a, b, c, d, e,f]],columns=name3)
    # df2转换成多项式
    po = PolynomialFeatures(degree=2, interaction_only=False, include_bias=False)
    df2_poly = po.fit_transform(df2)
    df2_dataframe = pd.DataFrame(df2_poly, )

    print("输入场次过程如下：")
    print(df3)
    m=input("请输入查找最近邻的数目：")

#计算相似度
    list5 = []
    df_all = pd.concat((df[name3].iloc[:248,:], df3), axis=0, join='outer')# axis=0表示纵轴，方向从上到下，体现出行的增加或减少，outer在列的方向上进行外连接（即求并集）

    # 归一化
    normal = preprocessing.MinMaxScaler()
    df_allnormal = normal.fit_transform(df_all)#Min-Max归一化
    # 与查找值进行相减，并返回绝对值
    df_nn = abs(df_allnormal - df_allnormal[-1])

    # 再对df_nn进行归一化,因为想把最不相似的上限定为6，然后面雨量累积乘以3，最大两日降雨降雨乘以2，最大三日降雨乘以2，
    # 增加权重之后，最不相似上限定为10
    df_nnnormal = normal.fit_transform(df_nn) # 先拟合数据，然后转化它将其转化为标准形式

    # 增加权重
    df_nnnormal[:, 0] = df_nnnormal[:, 0] * 3
    df_nnnormal[:, 2] = df_nnnormal[:, 2] * 2
    df_nnnormal[:, 3] = df_nnnormal[:, 3] * 2

    # 去除最后一行
    df_nnnormal = df_nnnormal[:-1, :]

    # 每一行求和
    df_nnnormal_sum = np.sum(df_nnnormal, axis=1)
    # 求最小的f个元素
    N_small = pd.DataFrame(df_nnnormal_sum).sort_values(by=0, ascending=True).head(int(m))#ascending=False表示降序排序，默认升序
    # 求最小的f个元素对应行号
    N_samll_index = pd.DataFrame({'相似度': df_nnnormal_sum}).sort_values(by='相似度', ascending=True)
    N_samll_index = list(N_samll_index.index)[:int(m)]

    df_sim=pd.DataFrame()
    for i in N_small.index:
        df_sim = pd.concat((df_sim, df.iloc[i:i + 1,:]), axis=0, join='outer')  # 0表示纵轴，方向从上到下，体现出行的增加或减少，outer在列的方向上进行外连接（即求并集）
    df_sim["相似度"] = N_small[0].tolist()#[0],DataFrame的列索引
    df_sim["name"] = N_samll_index
    df_sim=df_sim[["开始时间","name","相似度","累积雨量", "降雨历时", "最大48小时降雨", "万三累积雨量", '前10天降雨量', "起涨流量","最大洪峰"]]
    df_sim=df_sim.set_index("开始时间")
    #保存小数点设置
    list6 = ["相似度"]
    for i in list6:
        df_sim[i] = df_sim[i].apply(lambda x: round(x, 4))#apply，自动遍历整个 Series 或者 DataFrame, 对每一个元素运行指定的函数。#四舍五入 2保留小数部分个数
    list7 = ["累积雨量", "最大48小时降雨", "万三累积雨量", "前10天降雨量"]
    for i in list7:
        df_sim[i] = df_sim[i].apply(lambda x: round(x, 1))
    list8 = ["降雨历时", "起涨流量","name", "最大洪峰"]
    for i in list8:
        df_sim[i] = df_sim[i].astype(int)


    # 保存查找结果为csv
    print("查找相似过程如下：")
    print(df_sim)
    df_sim.to_csv('查找结果/第%s次三峡区间查找结果.csv' % n,index = False)#index = False表示不写入索引

    x=input("您是否继续程序，如果结束程序请按0:")
    if x=='0':
        break
    print('____________\n')
    # # 保存查找结果为excel
    # print("查找相似过程如下：")
    # print(df_sim)
    # writer = pd.ExcelWriter('第%s次三峡区间查找结果.xlsx' % n)#保存多个sheet在同一工作簿中
    # df_sim.to_excel(writer, 'page_1', float_format='%.5f')  #sheet的名字page_1， float_format 控制精度
    # writer.save()
