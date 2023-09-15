import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime,timedelta
import matplotlib.dates as mdates
import matplotlib
import numpy as np
import cv2
import os
import skimage.io as io
import natsort
import shutil

# 字体设置，bold（粗体显示）
font = {'family': 'MicroSoft YaHei',
        'weight': 'bold',
         'size': 10}
# font,字体。设置字体（font）的更多属性
matplotlib.rc("font", **font)

# open(file, mode='r', encoding=None） file：指定一个将要打开的文件的路径（绝对路径或相对路径） ；mode：可选参数，指定文件的打开模式：
# encoding：可选参数，用于指定解码或编码文件的编码名称，该参数应该只在文本模式下使用
# m=open("三峡区间流量处理1.csv","r", encoding="UTF-8")
# UTF－8编码则是用以解决国际上字符的一种多字节编码，它对英文使用8位（即一个字节），中文使用24位（三个字节）来编码。
m = open("三峡区间流量处理2022.csv", "r", encoding="UTF-8")
# pandas读取.csv文件
df_data = pd.read_csv(m)

# 生成特定格式的时间特征
def strtime_to_datetime(list2):
    list1 = []
    for i in list2:
        # time1stamp=datetime.strptime(i,"%Y-%m-%d %H:%M")
        time1stamp = datetime.strptime(i,"%Y-%m-%d %H:%M:%S")
        list1.append(time1stamp)

    return list1

# 生成特定格式的时间特征
def strtime_to_datetime1(list2):
    list1 = []
    for i in list2:
        # 将list2中数据转化成字符串，然后转化成特定的日期格式
        time1stamp = datetime.strptime(str(i),"%Y-%m-%d %H:%M:%S")
        # print(type(i),i)
        list1.append(time1stamp)

    return list1

# 转换成特定格式的时间特征
df_data["time"] = strtime_to_datetime(df_data["time"])
# print(type(df_data["time"][0]))
# 重新设置df_data（DataFrame）的行索引，df_data是pandas的实例化的一个对象
df_data = df_data.set_index("time")
# 可以将缺失值填充为指定的值,True让填充马上生效
df_data.fillna(0, inplace=True)
# print(df_data)

# f=open("场次降雨径流划分.csv","r", encoding="UTF-8")
# gbk的中文编码是双字节来表示的，英文编码是用ASC||码表示的，既用单字节表示
f = open("场次降雨径流划分2022.csv","r", encoding="gbk")
# pandas读取.csv文件，同时实例化df_info为pandas的对象
df_info = pd.read_csv(f)

def plot(i,df_info,df_data):
# def plot(i, df_info):

    # 生产另一个y轴，坐标值倒叙
    # plt.twinx().invert_yaxis()# 共用x坐标轴
    # 画分割时间段的降雨
    star, end = df_info["开始时间"][i],df_info["结束时间"][i]

    df_info["开始时间"] = strtime_to_datetime1(df_info["开始时间"])
    df_info["结束时间"] = strtime_to_datetime1(df_info["结束时间"])
    # timedelta直接实例化时间差
    df_info["取数开始时间"] = df_info["开始时间"] - timedelta(hours=24)
    df_info["取数结束时间"] = df_info["结束时间"] + timedelta(hours=24)
    time_star, time_end = df_info["取数开始时间"][i], df_info["取数结束时间"][i]

    df_plotyu = df_data[time_star:time_end]
    # 画正常柱状图;align：指定x轴上对齐方式，"center","lege"边缘
    plt.bar(df_plotyu.index, df_plotyu["P"], width=0.03, color="deepskyblue", align='edge')
    # df_plotyu_p=df_data[star:end]
    # 画正常柱状图
    # plt.bar(df_plotyu_p.index, df_plotyu_p["P"], width=0.03, color='gold' , align='edge')

    # plt.legend(['降雨','划分降雨'],loc='lower right')；给图加上图例；upper left
    # 配置横坐标为日期格式,plt.gca( )进行坐标轴的移动
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H'))
    # 指定时间点显示(1,5,9,13，17，21)，如果按月显示则改为MonthLocator()
    plt.gca().xaxis.set_major_locator(mdates.HourLocator(interval=4))
    # 自动旋转日期标记
    plt.gcf().autofmt_xdate()
    # 限制轴的范围
    plt.ylim((0, 6))

    # 保存
    fname = "第%s场次降雨径流%s%s" % (i, df_info["最大洪峰时间"][i].split(':')[0], df_info["峰型"][i])
    path = "场次降雨原始图片\\"
    print(fname)

    # 检查目录是否存在
    if not os.path.exists(path):
        # 如果不存在则创建目录
        os.makedirs(path)
    # 保存图片，命名图片
    plt.savefig(os.path.join(path, fname))
    # 绘图显示同时继续跑下面的代码
    plt.show(block=False)
    # 需要重新更新画布，否则会出现同一张画布上绘制多张图片
    plt.clf()

# 调用画图函数
for i in range(len(df_info)):
    plot(i, df_info, df_data)


# 图片裁剪
path = input("请输入降雨量数据文件名：")
path1 = input("请输入特征提取后保存的文件夹：")

# 创建预处理文件夹
if os.path.exists(path1):
    # print('true')
    # os.rmdir(file_dir)
    # 删除再建立
    shutil.rmtree(path1)
    os.makedirs(path1)
else:
    os.makedirs(path1)

# 批量读取图片
coll = io.ImageCollection(path+'/*.png')

# 遍历该目录下的所有图片文件
def read_path(file_pathname):
    # 传入相应的路径，将会返回那个目录下的所有文件名
    file_list = os.listdir(file_pathname)
    # 引入第三方库排序，sort无法得到想要的结果
    file_list1 = natsort.natsorted(file_list)
    for filename in file_list1:
        # print(filename)
        # #np.fromfile从磁盘读取原始数据，imdecode从内存中的缓冲区读取图像，-1读入完整图片，如果0读入灰度图片，1读一幅彩图
        src = cv2.imdecode(np.fromfile(file_pathname+'/'+filename, dtype=np.uint8), -1)
        # 选定想要的区域，h,w
        src1 = src[60:385, 80:570]
        # 选定想要的区域，h,w
        # src1 = src[200:1100, 300:1700]
        # tofile,将数组中的数据以二进制格式写进文件
        cv2.imencode('.png', src1)[1].tofile(path1 + '/' + filename + '.png')
read_path(path)


# 重命名图片
def myrename(path):
    file_list = os.listdir(path)
    # 引入第三方库排序，sort无法得到想要的结果
    file_list1 = natsort.natsorted(file_list)
    # print(file_list1)
    i = 0
    for fi in file_list1:
        # 函数用于路径拼接文件路径，可以传入多个路径进行拼接
        old_name = os.path.join(path, fi)
        new_name = os.path.join(path, str(i) + ".png")
        # 用于重命名文件或目录
        os.rename(old_name, new_name)
        i += 1

if __name__ == '__main__':
    myrename(path1)

# 批量生成文件夹
for i in range(99,262):
    name = 'S' + '%s' % (i + 1)
    # 生成文件夹
    os.makedirs(name)
    # 移动文件夹到data文件里
    shutil.move('%s'%name, 'data')

# 按顺序批量复制指定文件夹下的文件到另外一个指定文件夹下
for i in range(99,262):
    shutil.copy('色斑图修改\\%s.png'%i, 'data\\S%s'%(i+1))
