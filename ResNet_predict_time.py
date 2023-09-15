# -*- coding:utf-8 -*-
# coding: utf-8
# from IPython import get_ipython
# get_ipython().run_line_magic('matplotlib', 'inline')
import shutil
from torchvision import transforms, models, datasets
import torchvision.datasets as dset
from torch.utils.data import DataLoader, Dataset
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
import csv

'''这段代码定义了一个名为Config的类，该类具有一个类变量testing_dir，其值为字符串"色斑图数据集/test/"。类变量是指在类定义中定义的变量，它们是该类所有实例共享的变量。
该代码的作用可能是为了在一个程序中存储测试数据集的路径，以便于在代码中的其他位置使用该路径。使用类变量的好处是可以通过类名访问该变量，而不必创建类的实例，因为它是与类关联的。'''


# 配置
class Config:
    # testing_dir = "./data/faces/testing/"
    # testing_space_dir = "色斑图数据集/picture/"
    testing_time_dir = "哈希场次降雨图片/"


'''  定义了一个名为"SiameseNetwork"的类，继承自"nn.Module"类（nn由torch中导出）。
在类的构造函数中，使用"super"函数调用父类的构造函数，并将"self"作为第一个参数传递。这里没有进行任何额外的初始化操作。
类中定义了一个名为"forward_once"的函数，该函数将输入数据"X"作为参数传递，并通过调用另一个名为"model_ft"的模型对其进行前向传递（即推理）。最后，将输出结果返回。
类中还定义了一个名为"forward"的函数，该函数接受两个输入参数"input1"和"input2"，分别将它们作为输入传递给"forward_once"函数进行前向传递。
最后，将两个输出结果返回。这个类的作用是用于实现孪生神经网络的前向传递。'''


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


# 展示图像的函数
def imshow(img, m, x_i_dir, x_m_dir, path, text=None, should_save=False):
    # Tensor变量转换为ndarray变量
    # npimg = img.numpy()
    npimg = img.cpu().numpy()
    # 关闭坐标轴
    plt.axis("off")
    if text:
        # bold 粗体；alpha，透明度0至1
        plt.text(75, 8, text, style='italic', fontweight='bold',
                 bbox={'facecolor': 'white', 'alpha': 0.8, 'pad': 10})
    # 进行格式的转换
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    # 保存；.split()：拆分字符串。通过指定分隔符对字符串进行切片，并返回分割后的字符串列表（list）
    x_i_dir = x_i_dir.split('.')[0]
    x_m_dir = x_m_dir.split('.')[0]
    fname = "第%s相似%s_%s" % (m + 1, x_i_dir, x_m_dir)
    # 检查目录是否存在
    if not os.path.exists(path):
        # 如果不存在则创建目录
        os.makedirs(path)
    # 保存图片，命名图片
    plt.savefig(os.path.join(path, fname))
    # 显示所画的图
    # plt.show()


'''
当我们进行特征提取时，此辅助函数将模型中参数的 .requires_grad 属性设置为False。
默认情况下，当我们加载一个预训练模型时，所有参数都是 .requires_grad = True，
如果我们从头开始训练或微调，这种设置就没问题。但是，如果我们要运行特征提取并且只想为新初始化的层计算梯度，
那么我们希望所有其他参数不需要梯度变化。
'''


# 设置所需参数
def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        # model.parameters()保存的是Weights和Bais参数的值
        for param in model.parameters():
            # 梯度更新改为False，相当于冻住,模型（resnet）的参数不更新
            param.requires_grad = False


# 是否用人家训练好的特征来做，是（初始化模型）
def initialize_model(num_classes, feature_extract, use_pretrained=True):
    # 下面是自动下载的resnet的代码，加载预训练网络resnet152-b121ed2d.pth
    # model_ft = torchvision.models.resnet152()
    # model_ft.load_state_dict(torch.load('./resnet152-b121ed2d.pth'))
    # model_ft = torchvision.models.resnet152(pretrained=True)
    model_ft = models.resnet152(pretrained=use_pretrained)
    # 将特征提取的模块冻住，只训练FC层
    set_parameter_requires_grad(model_ft, feature_extract)
    # 获取全连接层输入特征
    num_ft_rs = model_ft.fc.in_features
    # 重新加全连接层，重新设置，使用PyTorch修改了一个模型（model_ft）的最后一个全连接层（fc），将其替换为一个新的全连接层，新的全连接层由nn.Sequential类创建。
    # nn.Sequential是一个容器类，允许你创建一个由多个网络层组成的序列
    # 在这里，它包含了一个线性层(nn.Linear)，输入特征数为num_ft_rs，输出特征数为num_classes，即将原始模型的输出特征数从num_ft_rs改为了num_classes）
    model_ft.fc = nn.Sequential(nn.Linear(num_ft_rs, num_classes))  # dim=0表示对列运算（1是对行运算），且元素和为1；

    return model_ft


# 定义精度评价函数
def evaluation(y_true, y_predict):
    r = np.corrcoef(y_true, y_predict)[0, 1]
    mse = np.mean((y_true - y_predict) ** 2)
    rse = mse / np.mean(y_true ** 2)
    return r, mse, rse


os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# feature_extract：True False 选择是否冻结参数 若是True 则冻住参数 反之不冻住参数
feature_extract = True
# cuda是否可用
train_on_gpu = torch.cuda.is_available()
if not train_on_gpu:
    print('CUDA is not available.  Training on CPU...')
else:
    print('CUDA is available!  Training on GPU...')

# torch.device代表将torch.Tensor分配到的设备的对象
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# feature_extract：True False 选择是否冻结参数 若是True 则冻住参数 反之不冻住参数
feature_extract = True

# 函数initialize_model()初始化一个用于迁移学习的神经网络模型
# 50：指定模型将分类的类别数量。这意味着模型的输出层将有50个节点，每个节点对应一个类别。
# feature_extract：一个布尔值变量，指定是否要对预训练的模型进行微调。(选择保存模型参数，并且冻结参数选择不更新)
# use_pretrained=True：一个布尔值变量，指定是否要使用预训练的权重来初始化模型。如果设置为True，将加载预训练的模型权重，否则将随机初始化模型的权重。（Ture）
model_ft = initialize_model(200, feature_extract, use_pretrained=True)

# GPU计算
model_ft = model_ft.to(device)

'''路径修改'''
# 定义了一个字符串变量path，其中存储了一个文件路径字符串'模型参数/checkpoint_similarity_space.pth'。这个路径指向了一个存储了模型参数的文件。
path = '../project/项目工程文件/LSTM_Siamese_project_file/数据集制作及模型参数训练/色斑图模型参数训练程序/checkpoint_similarity_time(02_200_224).pth'
# model_ft.load_state_dict(torch.load(path)['models'])

# 调用了PyTorch中的load_state_dict()方法来加载模型参数。model_ft是一个PyTorch模型，它将会被加载模型参数。
# 使用了torch.load()方法来加载保存在文件中的模型参数，其中map_location=torch.device('cpu')是一个可选参数，它告诉PyTorch将模型参数加载到CPU内存中。
# 该方法返回了一个字典对象，其中存储了加载的模型参数。字典中包含了'models'键和它对应的值，这个值就是保存的模型参数。使用['models']索引操作获取这个值，并将其赋给model_ft对象的参数中，这样就完成了模型参数的加载。
model_ft.load_state_dict(torch.load(path, map_location=torch.device('cpu'))['models'])

# net = SiameseNetwork().cuda()
# 前向传播的孪生网络
net = SiameseNetwork()

# transforms.Resize((100, 100))：该操作将输入图像的大小调整为(100, 100)像素大小，这可以用来使不同尺寸的图像具有相同的大小，以便输入到深度学习模型中。
# transforms.ToTensor()：该操作将输入的图像数据转换为张量格式，这是深度学习模型处理图像数据所需的格式。
# 这些操作被组合在一起使用，可以通过调用transforms.Compose()来实现，这个函数将多个transforms操作串联在一起形成一个操作序列，它被存储在一个名为"photo_transforms"的变量
photo_transforms = transforms.Compose([transforms.Resize((100, 100)),
                                       transforms.ToTensor()
                                       ])

# 读取场次降雨径流划分文件
path = os.path.dirname(os.path.realpath(__file__)) + '\\原始数据\\场次降雨径流划分.csv'
# os.path.dirname(os.path.realpath(__file__)),获得你刚才所引用的模块所在的绝对路径，__file__为内置属性。
df = pd.read_csv(path, encoding='UTF-8')
df = df[["开始时间", "累积雨量", "降雨历时", "最大小时降雨", "万三累积雨量", '前10天降雨量', "起涨流量", "最大洪峰"]]

'''路径修改'''
# 删除文件夹下的原图片
path = ['ResNet_image_time(02_200_224)_comparison\\', "figure_ResNet_time(02_200_224)_K\\"]

for n in range(2):
    if os.path.exists(path[n]):  # 检查目录是否存在
        shutil.rmtree(path[n])  # 删除该文件夹和文件夹下所有图片

# 按照8：2比例划查找集和测试集（210：52）
data_k = []
data_r = []
data_mse = []
data_rse = []
true = []
predict1, predict2, predict3, predict4, predict5, predict6, predict7, predict8, predict9, predict10 = [], [], [], [], [], [], [], [], [], []
print('开始测试......')
for i in range(210, 262):
    print('测试样本为{}'.format(i))
    list1 = []
    name = []
    distance = []
    true.append(df.loc[i]['最大洪峰'])
    x_i_dir = f'{i}.png'
    # x_i_0 = Image.open(os.path.join(Config.testing_space_dir, x_i_dir))  # 读取指定图片
    x_i_1 = Image.open(os.path.join(Config.testing_time_dir, x_i_dir))

    # x_i_0 = x_i_0.convert("RGB")  # RGBA转换为RGB图像，原png格式图片4通道位数为32位，转换完成为jpg格式的3通道为24位
    # x_i_0 = photo_transforms(x_i_0)  # 调用
    # x_i_0 = x_i_0.unsqueeze(0)  # 在第一维的左边插入一个维度
    # print(type(x0),x0.shape)

    # 代码使用了PyTorch深度学习库中的.cuda()方法，它是将数据从CPU（中央处理器）移动到GPU（图形处理器）的方法。具体而言，这行代码将一个名为x0的变量从CPU上的主存（RAM）移动到GPU上的显存（VRAM）。
    # 在深度学习中，使用GPU来进行计算可以加快模型的训练和推断速度，因为GPU有更多的并行处理能力。因此，通常会将数据从CPU移动到GPU上进行处理，然后将结果返回到CPU进行下一步操作
    # x_i_0 = x_i_0.cuda()

    x_i_1 = x_i_1.convert("RGB")
    x_i_1 = photo_transforms(x_i_1)
    x_i_1 = x_i_1.unsqueeze(0)
    x_i_1 = x_i_1.cuda()

    # df_data = pd.read_csv('.\\查找结果\\第1次三峡区间查找结果.csv')
    # 保存与指定图片的不相似度数据
    for g in range(0, 210):
        x_g_dir = f'{g}.png'
        # x_g_0 = Image.open(os.path.join(Config.testing_space_dir, x_g_dir))  # 读取指定图片
        x_g_1 = Image.open(os.path.join(Config.testing_time_dir, x_g_dir))

        # x_g_0 = x_g_0.convert("RGB")  # RGBA转换为RGB图像，原png格式图片4通道位数为32位，转换完成为jpg格式的3通道为24位
        # x_g_0 = photo_transforms(x_g_0)
        # x_g_0 = x_g_0.unsqueeze(0)  # 在第一维的左边插入一个维度
        # x_g_0 = x_g_0.cuda()

        x_g_1 = x_g_1.convert("RGB")  # RGBA转换为RGB图像，原png格式图片4通道位数为32位，转换完成为jpg格式的3通道为24位
        x_g_1 = photo_transforms(x_g_1)
        x_g_1 = x_g_1.unsqueeze(0)  # 在第一维的左边插入一个维度
        x_g_1 = x_g_1.cuda()

        '''将输入的两个图像张量 x0 和 x1 封装为 PyTorch 的 Variable 类型的对象，然后将它们作为参数传递给名为 net 的神经网络
            这个神经网络 net 的输出包含两个部分，分别被赋值给 output1 和 output2 两个变量。这两个变量分别代表着对应于输入图像 x0 和 x1 的特征向量
            接下来，使用 PyTorch 的 F.pairwise_distance 函数计算了 output1 和 output2 之间的欧几里得距离，并将结果赋值给 euclidean_distance 变量
            因此，这段代码的作用是将两个输入图像 x0 和 x1 送入神经网络 net 进行特征提取，然后计算这两个特征向量之间的欧几里得距离。这通常用于度量两个图像的相似性或差异性'''
        # output1_0, output2_0 = net(Variable(x_i_0), Variable(x_g_0))
        output1_1, output2_1 = net(Variable(x_i_1), Variable(x_g_1))
        # euclidean_space_distance = F.pairwise_distance(output1_0, output2_0)  # 计算特征图之间的像素级的距离
        euclidean_time_distance = F.pairwise_distance(output1_1, output2_1)
        x_without_ext, ext = os.path.splitext(x_g_dir)
        name.append(x_without_ext)
        distance.append('{:.6f}'.format(euclidean_time_distance.item()))

    # 将三个列表name和distance中的元素按照索引位置一一对应，打包成一个元组，并将所有的元组组成一个新的列表z
    z = list(zip(name, distance))
    # 注意append和extend的区别:.append()方法用于向列表末尾添加单个元素，该元素可以是任意数据类型;
    # .extend()方法用于将一个可迭代对象中的所有元素添加到列表末尾，该可迭代对象可以是一个列表、元组、集合或字符串等
    list1.extend(z)
    dissimilarity = np.mat(list1)  # 转化为矩阵
    a = ["name", "dissimilarity"]
    a = np.mat(a)
    dissimilarity = np.r_[a, dissimilarity]  # 把对应的名称加上
    # np.savetxt('查找结果/dissimilarity1.csv', dissimilarity, fmt='%s', delimiter=',')
    data = pd.DataFrame(dissimilarity)
    data.columns = data.iloc[0]
    data = data.iloc[1:]
    # print(data)
    for k in range(1, 11):
        print('当K={a}时，计算预测精度......'.format(a=k))
        # print(data)
        # 排序后保存最相似的n个数据
        # df = pd.read_csv('查找结果/dissimilarity1.csv')
        dissimilafirity_sort = data.sort_values(by='dissimilarity')  # ascending=False表示降序排序，默认升序
        dissimilafirity_sort_index = list(dissimilafirity_sort['name'])[:int(k)]  # dissimilafirity_sort_index是列表格式
        dissimilafirity_sort_index = [int(x) for x in dissimilafirity_sort_index]
        # print(dissimilafirity_sort)
        # print(dissimilafirity_sort_index)
        # 将字符串数据逐个转换为整型
        # print(dissimilafirity_sort)
        # print(dissimilafirity_sort_index)
        # dissimilafirity_sort = df
        # print(dissimilafirity_sort)
        # dissimilafirity_sort[0:10].to_csv('查找结果/dissimilarity_sort1.csv', index=False)  # index = False表示不写入索引

        '''相似图像对照'''
        if k < 2:
            for m in range(3):
                x_m_dir = dissimilafirity_sort.iloc[m, 0]
                x_m_dir = f'{x_m_dir}.png'
                dissimilarity = float(dissimilafirity_sort.iloc[m, 1])
                # x_m_0 = Image.open(os.path.join(Config.testing_space_dir, x_m_dir))  # 读取指定图片
                x_m_1 = Image.open(os.path.join(Config.testing_time_dir, x_m_dir))

                # x_m_0 = x_m_0.convert("RGB")  # RGBA转换为RGB图像，原png格式图片4通道位数为32位，转换完成为jpg格式的3通道为24位
                # x_m_0 = photo_transforms(x_m_0)
                # x_m_0 = x_m_0.unsqueeze(0)  # 在第一维的左边插入一个维度
                # x_m_0 = x_m_0.cuda()
                x_m_1 = x_m_1.convert("RGB")  # RGBA转换为RGB图像，原png格式图片4通道位数为32位，转换完成为jpg格式的3通道为24位
                x_m_1 = photo_transforms(x_m_1)
                x_m_1 = x_m_1.unsqueeze(0)  # 在第一维的左边插入一个维度
                x_m_1 = x_m_1.cuda()

                # concatenated_0 = torch.cat((x_i_0, x_m_0), 0)  # 在给定维度上对输入的张量序列进行连接操作
                concatenated_1 = torch.cat((x_i_1, x_m_1), 0)
                x_i_dir = x_i_dir.split('.')[0]
                # path_0 = "ResNet_image_comparison\\space\\{}".format(x_i_dir)

                '''路径修改'''
                path_1 = "ResNet_image_time(02_200_224)_comparison\\{}".format(x_i_dir)

                # imshow(torchvision.utils.make_grid(concatenated_0), m, x_i_dir, x_m_dir, path_0, '{}_Dissimilarity: {:.4f}'.format(m + 1, dissimilarity))  # 第二项对应imshow函数的text,即把不相似度画在图上
                imshow(torchvision.utils.make_grid(concatenated_1), m, x_i_dir, x_m_dir, path_1, '{}_Dissimilarity: {:.4f}'.format(m + 1, dissimilarity))

        '''获取预测洪峰'''
        df_sim = pd.DataFrame()
        for j in dissimilafirity_sort_index:
            # 0表示纵轴，方向从上到下，体现出行的增加或减少，outer在列的方向上进行外连接（即求并集）
            df_sim = pd.concat((df_sim, df.iloc[j:j + 1, 7:8]), axis=0, join='outer')
        # print(df_sim)
        df_sim_value = round(df_sim['最大洪峰'].mean(), 2)
        # print(df_sim_value)
        # print(df.loc[i]['最大洪峰'])
        # print('预测的平均最大洪峰为：', df_sim_value)
        if k == 1:
            predict1.append(df_sim_value)
        elif k == 2:
            predict2.append(df_sim_value)
        elif k == 3:
            predict3.append(df_sim_value)
        elif k == 4:
            predict4.append(df_sim_value)
        elif k == 5:
            predict5.append(df_sim_value)
        elif k == 6:
            predict6.append(df_sim_value)
        elif k == 7:
            predict7.append(df_sim_value)
        elif k == 8:
            predict8.append(df_sim_value)
        elif k == 9:
            predict9.append(df_sim_value)
        elif k == 10:
            predict10.append(df_sim_value)

'''绘制散点图'''
# 清除之前的绘图
predict = [predict1, predict2, predict3, predict4, predict5, predict6, predict7, predict8, predict9, predict10]
for k in range(0, 10):
    plt.clf()
    plt.scatter(true, predict[k])
    # 绘制拟合线
    # sns.regplot(x=true, y=predict)
    # 绘制参考线
    plt.plot([0, 30000], [0, 30000], '--')
    # 添加标签和标题
    plt.xlabel('True Values')
    plt.ylabel('Predictions')
    plt.title('K={a},True vs. Predicted'.format(a=k+1))

    '''路径修改'''
    # 指定图片保存路径
    figure_save_path = "figure_ResNet_time(02_200_224)_K"

    if not os.path.exists(figure_save_path):
        os.makedirs(figure_save_path)  # 如果不存在目录figure_save_path，则创建
    plt.savefig(os.path.join(figure_save_path, 'k={a}.png'.format(a=k+1)))  # 第一个是指存储路径，第二个是图片名字
    # # 显示图像
    # plt.show()

    '''评价函数'''
    # 将列表转化为元组形式
    y_true = np.array(tuple(true))
    y_predict = np.array(tuple(predict[k]))
    r, mse, rse = evaluation(y_true, y_predict)
    # noinspection PyUnboundLocalVariable
    data_k.append(k+1)
    data_r.append(round(r, 3))
    data_mse.append(round(int(mse)))
    data_rse.append(round(rse, 3))

print('开始保存所有精度评价分数......')

'''路径修改'''
# 创建文件对象
data = open('评价结果/K_ResNet_time(02_200_224)_evaluation.csv', 'w', encoding='utf-8', newline='')

#  基于文件对象构建 csv写入对象
csv_writer = csv.writer(data)
# 构建列表头
csv_writer.writerow(['K值', "相关系数", "均方误差", "相对平方误差"])
data.close()

'''路径修改'''
# 填入数据
data = pd.read_csv('评价结果/K_ResNet_time(02_200_224)_evaluation.csv')

data["K值"] = data_k
data["相关系数"] = data_r
data["均方误差"] = data_mse
data["相对平方误差"] = data_rse

'''路径修改'''
data.to_csv('评价结果/K_ResNet_time(02_200_224)_evaluation.csv', index=False)