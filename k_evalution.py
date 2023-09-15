from scipy.interpolate import make_interp_spline
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib

font = {'family': 'MicroSoft YaHei'}
matplotlib.rc("font", **font)  # win可以显示中文

# data = pd.read_csv('评价结果/K_evaluation.csv')
# data = pd.read_csv('评价结果/K_ResNet_evaluation.csv')
# data = pd.read_csv('评价结果/K_lasso360_evaluation.csv')
# data = pd.read_csv('评价结果/K_ResNet_time_evaluation.csv')
# data = pd.read_csv('评价结果/K_ResNet_space_evaluation.csv')
data = pd.read_csv('评价结果/K_haxisuanfa_evaluation.csv')
x = data['K值']
Y = ['相关系数', '相对平方误差']
for i in Y:
    y = data[i]
# y = data['均方误差']
# y = data['相对平方误差']
# # 绘制折线图
    plt.plot(x, y)

# 绘制曲线图（进行插值操作）
# x_smooth = np.linspace(x.min(), x.max(), 300)
# y_smooth = make_interp_spline(x, y)(x_smooth)
# plt.plot(x_smooth, y_smooth)

    plt.xlabel('K值')
    plt.ylabel(i)
    plt.title('{}随K值变化趋势'.format(i))
    plt.xticks(range(0, 11, 1))
    plt.show()
