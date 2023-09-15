# -*- coding:utf-8 -*-
#from IPython import get_ipython
#get_ipython().run_line_magic('matplotlib', 'inline')
import shutil

from torchvision import transforms, models, datasets
import torchvision.datasets as dset
from torch.utils.data import DataLoader,Dataset
import matplotlib.pyplot as plt
import torchvision.utils
import numpy as np
import pandas as pd
import random
from PIL import Image
import torch
from torch.autograd import Variable
import PIL.ImageOps
import torch.nn as nn
import torch.nn.functional as F
import os


#展示图像的函数
def imshow(img, i, x_x, x, text=None,should_save=False):
    # npimg = img.numpy()#Tensor变量转换为ndarray变量
    npimg = img.cpu().numpy()
    plt.axis("off")#关闭坐标轴
    if text:
        plt.text(75, 8, text, style='italic',fontweight='bold',#bold 粗体
            bbox={'facecolor':'white', 'alpha':0.8, 'pad':10})#alpha，透明度0至1
    plt.imshow(np.transpose(npimg, (1, 2, 0)))#进行格式的转换
    #保存
    x_x = x_x.split('\\')[1].split('.')[0]
    x = x.split('\\')[1].split('.')[0]
    fname = "第%s相似%s_%sdata2" % (i+1,x_x,x)
    path = "色斑相似度图片\\"
    if not os.path.exists(path):  # 检查目录是否存在
        os.makedirs(path)  # 如果不存在则创建目录
    plt.savefig(os.path.join(path, fname))# 保存图片，命名图片
    # plt.show()#显示所画的图


class Config():#配置
    # testing_dir = "./data/faces/testing/"
    testing_dir = "色斑图数据集/test/"


feature_extract = True #feature_extract：True False 选择是否冻结参数 若是True 则冻住参数 反之不冻住参数

train_on_gpu = torch.cuda.is_available()
if not train_on_gpu:
    print('CUDA is not available.  Training on CPU...')
else:
    print('CUDA is available!  Training on GPU...')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():# model.parameters()保存的是Weights和Bais参数的值。
            param.requires_grad = False#梯度更新改为False，相当于冻住,模型（resnet）的参数不更新
'''
# 当我们进行特征提取时，此辅助函数将模型中参数的 .requires_grad 属性设置为False。
# 默认情况下，当我们加载一个预训练模型时，所有参数都是 .requires_grad = True，
# 如果我们从头开始训练或微调，这种设置就没问题。但是，如果我们要运行特征提取并且只想为新初始化的层计算梯度，
# 那么我们希望所有其他参数不需要梯度变化。
'''

def initialize_model( num_classes, feature_extract, use_pretrained=True):#是否用人家训练好的特征来做，是
    # 下面是自动下载的resnet的代码，加载预训练网络
    model_ft = models.resnet152(pretrained=use_pretrained)
    # 是否将特征提取的模块冻住，只训练FC层
    set_parameter_requires_grad(model_ft, feature_extract)
    # 获取全连接层输入特征
    num_ft_rs = model_ft.fc.in_features
    # 重新加全连接层，重新设置
    model_ft.fc = nn.Sequential(nn.Linear(num_ft_rs, num_classes))#dim=0表示对列运算（1是对行运算），且元素和为1；

    return model_ft

model_ft = initialize_model(50, feature_extract, use_pretrained=True)
model_ft = model_ft.to(device)#GPU计算


path = '模型参数/checkpoint_similarity.pth'
model_ft.load_state_dict(torch.load(path)['models'])


class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()

    def forward_once(self, x):
        output = model_ft(x)
        return output

    def forward(self, input1, input2):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        return output1, output2



# net = SiameseNetwork().cuda()
net = SiameseNetwork()


photo_transforms =transforms.Compose([transforms.Resize((100, 100)),
                                    transforms.ToTensor()
                                    ])
# x_x = input('请输入您想比较相似度的图片（类\图片名.后缀）：')#S201\200.png,S255\254.png
x_x = input('请输入您想比较相似度的图片：')
x_x = 'S%s\\%s.png'%(int(x_x)+1,x_x)
x0 = Image.open(os.path.join(Config.testing_dir, x_x))#读取指定图片
x0 = x0.convert("RGB")#RGBA转换为RGB图像，原png格式图片4通道位数为32位，转换完成为jpg格式的3通道为24位
x0 = photo_transforms(x0)
x0 = x0.unsqueeze(0)#在第一维的左边插入一个维度
# print(type(x0),x0.shape)
x0 = x0.cuda()


df_data = pd.read_csv('..\\第1次三峡区间查找结果.csv')
list3 = list(df_data['name'])
print(list3)
#保存与指定图片的不相似度数据
list1 = []
for i in list3:
    # if i == 0:
    #     continue
    list2 = []
    for j in range(1):
        if(j==0):
            if(i<9):
                x = 'S00%s\%s.png'%(i + 1,i)
            elif(i<99):
                x = 'S0%s\%s.png' % (i + 1, i)
            else:
                x = 'S%s\%s.png' % (i + 1, i)
        else:
            x = 'S%s\%s_%s.png' % (i + 1,i,j)
        x1 = Image.open(os.path.join(Config.testing_dir, x))  # 读取指定图片
        x1 = x1.convert("RGB")  # RGBA转换为RGB图像，原png格式图片4通道位数为32位，转换完成为jpg格式的3通道为24位
        x1 = photo_transforms(x1)
        x1 = x1.unsqueeze(0)  # 在第一维的左边插入一个维度
        # print(type(x1), x1.shape)
        x1 = x1.cuda()
        output1, output2 = net(Variable(x0), Variable(x1))
        euclidean_distance = F.pairwise_distance(output1, output2)#计算特征图之间的像素级的距离
        if(j==0):
            x = '%s.png'%(i)
        else:
            x = '%s_%s.png' % (i,j)
        S = []
        name = []
        distance = []
        if (i < 9):
            S.append('S00%s' % (i + 1))
        elif (i < 99):
            S.append('S0%s' % (i + 1))
        else:
            S.append('S%s' % (i + 1))
        name.append(x)
        distance.append('{:.6f}'.format(euclidean_distance.item()))
        z = list(zip(S,name,distance))
        print(z)
        list2.extend(z)#注意append和extend的区别，
    list1.extend(list2)

dissimilarity=np.mat(list1)#转化为矩阵
a = ["S","name","dissimilarity"]
a = np.mat(a)
dissimilarity = np.r_[a, dissimilarity]  # 把对应的名称加上
np.savetxt('查找结果/dissimilarity1.csv', dissimilarity, fmt='%s', delimiter=',')

# 排序后保存最相似的n个数据
df = pd.read_csv('查找结果/dissimilarity1.csv')
# dissimilafirity_sort = df.sort_values(by='dissimilarity')#ascending=False表示降序排序，默认升序
dissimilafirity_sort = df
# print(dissimilafirity_sort)
dissimilafirity_sort[0:10].to_csv('查找结果/dissimilarity_sort1.csv',index = False)#index = False表示不写入索引

#出图
data = pd.read_csv("查找结果/dissimilarity_sort1.csv")
path = "色斑相似度图片\\"
if os.path.exists(path):  # 检查目录是否存在
    shutil.rmtree(path)  # 删除该文件夹和文件夹下所有图片
for i in range(5):
    data_1 = data.iloc[i:i+1,0]#切片操作
    data_1 = data_1.values#dataframe类型转成数组类型
    data_2 = data.iloc[i:i+1,1]#切片操作
    data_2 = data_2.values#dataframe类型转成数组类型
    data_3 = data.iloc[i:i+1,2]#切片操作
    data_3 = data_3.values#dataframe类型转成数组类型
    [x] = data_1 + '\\' + data_2
    [dissimilarity] = data_3
    x1 = Image.open(os.path.join(Config.testing_dir, x))#读取指定图片
    x1 = x1.convert("RGB")#RGBA转换为RGB图像，原png格式图片4通道位数为32位，转换完成为jpg格式的3通道为24位
    x1 = photo_transforms(x1)
    x1 = x1.unsqueeze(0)#在第一维的左边插入一个维度
    x1 = x1.cuda()
    concatenated = torch.cat((x0, x1), 0)  # 在给定维度上对输入的张量序列进行连接操作
    imshow(torchvision.utils.make_grid(concatenated), i, x_x, x, '{}_Dissimilarity: {:.4f}'.format(i+1,dissimilarity))#第二项对应imshow函数的text,即把不相似度画在图上
