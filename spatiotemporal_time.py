import shutil
from torchvision import transforms, models
import matplotlib.pyplot as plt
import torchvision.utils
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import os
import csv


class Config:
    testing_time_dir = "rainfall time series histograms/"


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


# Functions that show comparative images
def imshow(img, m, x_i_dir, x_m_dir, path, text=None, should_save=False):
    npimg = img.cpu().numpy()
    plt.axis("off")
    if text:
        plt.text(75, 8, text, style='italic', fontweight='bold',
                 bbox={'facecolor': 'white', 'alpha': 0.8, 'pad': 10})
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    x_i_dir = x_i_dir.split('.')[0]
    x_m_dir = x_m_dir.split('.')[0]
    fname = "The %sth is similar %s_%s" % (m + 1, x_i_dir, x_m_dir)
    if not os.path.exists(path):
        os.makedirs(path)
    plt.savefig(os.path.join(path, fname))


# Setting the required parameters
def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            # The gradient update is changed to False, which is equivalent to freezing, and the parameters of the model (resnet) are not updated.
            param.requires_grad = False


# Whether to do it with features trained by people, yes (initialize the model)
def initialize_model(num_classes, feature_extract, use_pretrained=True):
    model_ft = models.resnet152(pretrained=use_pretrained)
    # Freeze the module for feature extraction and train only the FC layer
    set_parameter_requires_grad(model_ft, feature_extract)
    # Getting fully connected layer input features
    num_ft_rs = model_ft.fc.in_features
    # Re-adding fully-connected layers, resetting, and using PyTorch to modify the last fully-connected layer (fc) of a model (model_ft) by replacing it with a new fully-connected layer created by the nn.Sequential class.
    # Here, it contains a linear layer (nn.Linear) with num_ft_rs as the number of input features and num_classes as the number of output features, i.e., the number of output features of the original model was changed from num_ft_rs to num_classes)
    model_ft.fc = nn.Sequential(nn.Linear(num_ft_rs, num_classes))
    return model_ft


# Define the accuracy evaluation function
def evaluation(y_true, y_predict):
    r = np.corrcoef(y_true, y_predict)[0, 1]
    rmse = np.sqrt(np.mean((y_true - y_predict) ** 2))
    return r, rmse


os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
feature_extract = True
# Is cuda available?
train_on_gpu = torch.cuda.is_available()
if not train_on_gpu:
    print('CUDA is not available.  Training on CPU...')
else:
    print('CUDA is available!  Training on GPU...')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
feature_extract = True

# The function initialize_model() initializes a neural network model for transfer learning
# 50: Specifies the number of categories that the model will categorize. This means that the output layer of the model will have 50 nodes, each corresponding to a category.
# feature_extract: a boolean variable specifying whether to fine-tune the pre-trained model. (The model parameters are selected to be saved and the freeze parameter is selected not to be updated)
# use_pretrained=True: a boolean-valued variable that specifies whether or not to use pre-trained weights to initialize the model. If set to True, pre-trained model weights will be loaded, otherwise the model's weights will be randomly initialized. (Ture)
model_ft = initialize_model(200, feature_extract, use_pretrained=True)

model_ft = model_ft.to(device)

'''Path modification'''
path = 'model training parameters/checkpoint_similarity_time.pth'
model_ft.load_state_dict(torch.load(path, map_location=torch.device('cpu'))['models'])

# SiameseNetworks for forward propagation
net = SiameseNetwork()
photo_transforms = transforms.Compose([transforms.Resize((100, 100)),
                                       transforms.ToTensor()
                                       ])
# Read the field rainfall runoff delineation file
df = pd.read_csv('raw data/divisions of rainfall runoff.csv')
df = df[["start time", "cumulative rainfall", "rainfall duration", "maximum 48-hour rainfall", "Wan San cumulative rainfall", 'rainfall in the previous 10 days', "rising flow", "flood peak"]]

if os.path.exists('spatiotemporal_image_time_comparison\\'):
    shutil.rmtree('spatiotemporal_image_time_comparison\\')

data_k = []
data_r = []
data_rmse = []
true = []
predict1, predict2, predict3, predict4, predict5,  = [], [], [], [], []
print('Start testing......')

for i in range(210, 262):
    print('The test sample is {}'.format(i))
    list1 = []
    name = []
    distance = []
    true.append(df.loc[i]['flood peak'])
    x_i_dir = f'{i}.png'
    x_i_1 = Image.open(os.path.join(Config.testing_time_dir, x_i_dir))
    x_i_1 = x_i_1.convert("RGB")
    x_i_1 = photo_transforms(x_i_1)
    x_i_1 = x_i_1.unsqueeze(0)
    x_i_1 = x_i_1.cuda()

    for g in range(0, 210):
        x_g_dir = f'{g}.png'
        x_g_1 = Image.open(os.path.join(Config.testing_time_dir, x_g_dir))
        x_g_1 = x_g_1.convert("RGB")
        x_g_1 = photo_transforms(x_g_1)
        x_g_1 = x_g_1.unsqueeze(0)
        x_g_1 = x_g_1.cuda()

        '''Wrap the two input image tensors x0 and x1 as PyTorch objects of type Variable and pass them as arguments to a neural network called net.
    The output of this neural network, net, consists of two parts that are assigned to two variables, output1 and output2. 
    These two variables represent the feature vectors corresponding to the input images x0 and x1, respectively.
    Next, the Euclidean distance between output1 and output2 is computed using PyTorch's F.pairwise_distance function and assigned to the euclidean_distance variable.
    So, what this code does is to feed two input images x0 and x1 into the neural network net for feature extraction, and then compute the Euclidean distance between these two feature vectors.
    This is often used to measure the similarity or difference between two images.'''
        output1_1, output2_1 = net(Variable(x_i_1), Variable(x_g_1))
        euclidean_time_distance = F.pairwise_distance(output1_1, output2_1)
        x_without_ext, ext = os.path.splitext(x_g_dir)
        name.append(x_without_ext)
        distance.append('{:.6f}'.format(euclidean_time_distance.item()))

    z = list(zip(name, distance))
    list1.extend(z)
    dissimilarity = np.mat(list1)
    a = ["name", "dissimilarity"]
    a = np.mat(a)
    dissimilarity = np.r_[a, dissimilarity]
    data = pd.DataFrame(dissimilarity)
    data.columns = data.iloc[0]
    data = data.iloc[1:]

    for k in range(1, 6):
        print('When K={a}, the prediction accuracy is calculated......'.format(a=k))
        dissimilafirity_sort = data.sort_values(by='dissimilarity')
        dissimilafirity_sort_index = list(dissimilafirity_sort['name'])[:int(k)]
        dissimilafirity_sort_index = [int(x) for x in dissimilafirity_sort_index]

        '''Comparison of similar images'''
        if k < 2:
            for m in range(3):
                x_m_dir = dissimilafirity_sort.iloc[m, 0]
                x_m_dir = f'{x_m_dir}.png'
                dissimilarity = float(dissimilafirity_sort.iloc[m, 1])
                x_m_1 = Image.open(os.path.join(Config.testing_time_dir, x_m_dir))
                x_m_1 = x_m_1.convert("RGB")
                x_m_1 = photo_transforms(x_m_1)
                x_m_1 = x_m_1.unsqueeze(0)
                x_m_1 = x_m_1.cuda()

                concatenated_1 = torch.cat((x_i_1, x_m_1), 0)
                x_i_dir = x_i_dir.split('.')[0]

                path_1 = "spatiotemporal_image_time_comparison\\{}".format(x_i_dir)
                imshow(torchvision.utils.make_grid(concatenated_1), m, x_i_dir, x_m_dir, path_1, '{}_Dissimilarity: {:.4f}'.format(m + 1, dissimilarity))

        '''Getting predicted flood peaks'''
        df_sim = pd.DataFrame()
        for j in dissimilafirity_sort_index:
            df_sim = pd.concat((df_sim, df.iloc[j:j + 1, 7:8]), axis=0, join='outer')
        df_sim_value = round(df_sim['flood peak'].mean(), 2)
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

predict = [predict1, predict2, predict3, predict4, predict5]
for k in range(0, 5):
    y_true = np.array(tuple(true))
    y_predict = np.array(tuple(predict[k]))
    r, rmse = evaluation(y_true, y_predict)
    data_k.append(k+1)
    data_r.append(round(r, 3))
    data_rmse.append(round(int(rmse)))
print('Start saving all accuracy evaluation scores......')

data = open('evaluation results/spatiotemporal_time_evaluation.csv', 'w', encoding='utf-8', newline='')
csv_writer = csv.writer(data)
csv_writer.writerow(['K', "correlation coefficient", "Root mean square error"])
data.close()

data = pd.read_csv('evaluation results/spatiotemporal_time_evaluation.csv')
data["K"] = data_k
data["correlation coefficient"] = data_r
data["Root mean square error"] = data_rmse
data.to_csv('evaluation results/spatiotemporal_time_evaluation.csv', index=False)