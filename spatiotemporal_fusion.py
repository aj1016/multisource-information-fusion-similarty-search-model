from torchvision import transforms, models
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import os


class Config:
    testing_space_dir = "rainfall spatial distribution color spot maps/"
    testing_time_dir = "rainfall time series histograms/"


class SiameseNetwork_space(nn.Module):
    def __init__(self):
        super(SiameseNetwork_space, self).__init__()

    def forward_once(self, x):
        output = model_ft_space(x)
        return output

    def forward(self, input1, input2):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        return output1, output2


class SiameseNetwork_time(nn.Module):
    def __init__(self):
        super(SiameseNetwork_time, self).__init__()

    def forward_once(self, x):
        output = model_ft_time(x)
        return output

    def forward(self, input1, input2):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        return output1, output2


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
    model_ft.fc = nn.Sequential(nn.Linear(num_ft_rs, num_classes))  # dim=0表示对列运算（1是对行运算），且元素和为1；
    return model_ft


os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# Select whether to freeze the parameter If True then freeze the parameter, otherwise do not freeze the parameter
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
model_ft_space = initialize_model(100, feature_extract, use_pretrained=True)
model_ft_time = initialize_model(200, feature_extract, use_pretrained=True)

model_ft_space = model_ft_space.to(device)
model_ft_time = model_ft_time.to(device)

path_space = 'model training parameters/checkpoint_similarity_space.pth'
path_time = 'model training parameters/checkpoint_similarity_time.pth'

model_ft_space.load_state_dict(torch.load(path_space, map_location=torch.device('cpu'))['models'])
model_ft_time.load_state_dict(torch.load(path_time, map_location=torch.device('cpu'))['models'])

# SiameseNetworks for forward propagation
net_space = SiameseNetwork_space()
net_time = SiameseNetwork_time()
photo_transforms = transforms.Compose([transforms.Resize((100, 100)),
                                       transforms.ToTensor()
                                       ])

print('Start testing......')
df_data = pd.read_csv('process data/multi-source_hydrology_fusion.csv')
k = 1
name = []
test_name = []
distance_space = []
distance_time = []
data = pd.DataFrame()
for n in range(0, len(df_data)):
    list1 = []
    i = list(df_data['name'])[n]
    x_i_dir = f'{i}.png'
    x_i_0 = Image.open(os.path.join(Config.testing_space_dir, x_i_dir))
    x_i_1 = Image.open(os.path.join(Config.testing_time_dir, x_i_dir))

    x_i_0 = x_i_0.convert("RGB")
    x_i_0 = photo_transforms(x_i_0)
    x_i_0 = x_i_0.unsqueeze(0)
    x_i_0 = x_i_0.cuda()
    x_i_1 = x_i_1.convert("RGB")
    x_i_1 = photo_transforms(x_i_1)
    x_i_1 = x_i_1.unsqueeze(0)
    x_i_1 = x_i_1.cuda()

    # Saves the dissimilarity data with the specified image
    g = list(df_data['test_name'])[n]
    x_g_dir = f'{g}.png'
    print('When test_name={a}, start calculating similarity.......'.format(a=x_g_dir))
    x_g_0 = Image.open(os.path.join(Config.testing_space_dir, x_g_dir))
    x_g_1 = Image.open(os.path.join(Config.testing_time_dir, x_g_dir))

    x_g_0 = x_g_0.convert("RGB")
    x_g_0 = photo_transforms(x_g_0)
    x_g_0 = x_g_0.unsqueeze(0)
    x_g_0 = x_g_0.cuda()

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
    output1_0, output2_0 = net_space(Variable(x_i_0), Variable(x_g_0))
    output1_1, output2_1 = net_time(Variable(x_i_1), Variable(x_g_1))
    euclidean_space_distance = F.pairwise_distance(output1_0, output2_0)
    euclidean_time_distance = F.pairwise_distance(output1_1, output2_1)
    x_without_ext, ext = os.path.splitext(x_i_dir)
    name.append(x_without_ext)
    x_without_ext, ext = os.path.splitext(x_g_dir)
    test_name.append(x_without_ext)
    distance_space.append('{:.6f}'.format(euclidean_space_distance.item()))
    distance_time.append('{:.6f}'.format(euclidean_time_distance.item()))

    z = list(zip(name, test_name, distance_space, distance_time))
    list1.extend(z)
    dissimilarity = np.mat(list1)
    a = ["name", 'test_name', "dissimilarity_space", "dissimilarity_time"]
    a = np.mat(a)
    dissimilarity = np.r_[a, dissimilarity]
    data = pd.DataFrame(dissimilarity)

print('Start saving the results......')
data.to_csv('process data/multi-source_spatiotemporal_fusion.csv', index=False, header=False)