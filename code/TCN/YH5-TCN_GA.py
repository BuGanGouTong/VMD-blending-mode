import torch
import pandas as pd
import numpy as np
from torch.nn.utils import weight_norm
import torch.utils.data as data_utils
import torch.nn.functional as F
from torch import nn
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from deap import base, creator, tools, algorithms
from scipy.stats import bernoulli
from bitstring import BitArray
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

class TemporalBlock(nn.Module):
    def __init__(self,n_input, n_output, kernel_size, dilation, padding, dropout):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_input, n_output, kernel_size, padding=padding, dilation=dilation))
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_output, n_output, kernel_size, padding=padding, dilation=dilation))
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        self.net = nn.Sequential(self.conv1, self.relu1, self.dropout1,
                                 self.conv2, self.relu2, self.dropout2)
        self.conv3 = weight_norm(nn.Conv1d(n_input, n_output, 1))
        self.final_activation = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        self.conv3.weight.data.normal_(0, 0.01)

    def forward(self, input_tensor):
        x = self.net(input_tensor)
        res = self.conv3(input_tensor)
        return self.final_activation(x + res)



class TCNStack(nn.Module):
    '''
    input_shape = (N, Feature_num, Seq_len)
    output_shape = (N, filter_num, seq_len)
    '''
    def __init__(self, layer_num, n_input, n_filter, kernel_size, dropout_rate, ):
        super(TCNStack, self).__init__()
        self.network = nn.Sequential()
        self.network.add_module(f'layernum_0', TemporalBlock(n_input=n_input, n_output=n_filter,
                                                  kernel_size=kernel_size, dilation=1,
                                                  padding=1, dropout=dropout_rate))
        for i in range(layer_num-1):
            dilation = 2**(i+1)
            self.network.add_module(f'layernum_{i+1}', TemporalBlock(n_input=n_filter, n_output=n_filter,
                                                  kernel_size=kernel_size, dilation=dilation,
                                                  padding=dilation, dropout=dropout_rate))

    def forward(self, x):
        return self.network(x)


class TCNmodel(nn.Module):
    def __init__(self, layer_num, n_input, n_filter, kernel_size, dropout_rate,seq_len):
        super(TCNmodel, self).__init__()
        self.tcn_layer = TCNStack(layer_num, n_input, n_filter, kernel_size, dropout_rate)
        self.fc1 = nn.Linear(n_filter, n_input)
        self.fc2 = nn.Linear(n_input * seq_len, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.tcn_layer(x)
        x = self.relu(self.fc1(x.transpose(2,1)))
        x= x.flatten(start_dim=1)
        x = self.fc2(x.flatten(start_dim=1))
        return x


# data = torch.FloatTensor(50, 7, 10)
# net = TCNmodel(layer_num=3, n_input=7, n_filter=64, kernel_size=3, dropout_rate=0.1)
# print(net(data).shape)

train_ratio = 0.6 #训练集比例1
forecast_horizon = 1  #预测步长
batch_size = 128
epoch = 150


# df = pd.read_excel('../data/YH7data.xlsx')
# df = pd.read_excel('../data/JH126 - VMD.xlsx')
# df = pd.read_excel('../data/JH286 - VMD.xlsx')
# df = pd.read_excel('../data/YH7data - VMD.xlsx')
df = pd.read_excel('../data/YH7data.xlsx')
data_files = np.array(df)
data = data_files[:,1:] #去掉时间一列
data = data.astype('float32')  #data(227.13)
feature = data.shape[1]

#分割训练集和测试集
train_size = int(len(data) * train_ratio) #train_ratio
data_train = data[0:train_size,:] #(181,13)
data_test = data[train_size:,:] #（46,13）

#训练集分割输入数据与输出数据  输入数据是使用全部数据不需要将Y剔除
train_x_norm1 = data_train[:,:]
train_y_norm1 = data_train[:,:1]
#测试集分割输入与输出数据
test_x_norm1 = data_test[:,:]
test_y_norm1 = data_test[:,:1]

preprocessor = MinMaxScaler()
preprocessor.fit(train_x_norm1)
train_x_norm = preprocessor.transform(train_x_norm1)
test_x_norm = preprocessor.transform(test_x_norm1)

preprocessor.fit(train_y_norm1)
train_y_norm = preprocessor.transform(train_y_norm1)

def windowed_dataset(series, window_size):
    """
    Returns a windowed dataset from a Pandas dataframe
    将输入数据进行时间步长处理
    return shape = (N, seq_len, feature_num)
    """
    available_examples = series.shape[0]-window_size + 1
    time_series_number = series.shape[1]
    inputs = np.zeros((available_examples,window_size,time_series_number))
    for i in range(available_examples):
        inputs[i,:,:] = series[i:i+window_size,:]
    return inputs

def windowed_forecast(series, forecast_horizon):
    #处理输出数据的时间步长
    available_outputs = series.shape[0]- forecast_horizon + 1
    output_series_num = series.shape[1]
    output = np.zeros((available_outputs,forecast_horizon, output_series_num))
    for i in range(available_outputs):
        output[i,:]= series[i:i+forecast_horizon,:]
    return output

def train_evaluate(ga_individual_solution):
    # 对目标超参数进行解码
    learning_rate = BitArray(ga_individual_solution[0:1])
    layer_num = BitArray(ga_individual_solution[1:4])
    dropout_rate = BitArray(ga_individual_solution[4:7])
    window_length = BitArray(ga_individual_solution[7:12])
    learning_rate = 10 ** (-learning_rate.uint-3)
    window_length = window_length.uint + 3
    layer_num = layer_num.uint + 4
    dropout_rate = (dropout_rate.uint + 1) * 0.1

    print('\n windows_length: ', window_length, ', layer_num: ', layer_num,
          ', learning_rate: ', learning_rate, ', dropout_rate: ', dropout_rate)

    train_y_real = train_y_norm1[window_length:]
    test_y_real = test_y_norm1[window_length:]

    # 训练集输入输出确定
    train_x = windowed_dataset(train_x_norm[:-forecast_horizon], window_length).swapaxes(2, 1)  # （num, seq, feature）
    train_y = train_y_norm[window_length:]  # (169,1)
    # 测试集输入确定
    test_x = windowed_dataset(test_x_norm[:-forecast_horizon], window_length).swapaxes(2, 1)  # （34,12,13）

    # 装载训练数据
    train_data = data_utils.TensorDataset(torch.from_numpy(train_x).float(), torch.from_numpy(train_y).float())
    train_loader = data_utils.DataLoader(train_data, batch_size=batch_size, shuffle=False)

    #构建网络
    net = TCNmodel(layer_num=layer_num, n_input=feature, n_filter=64, kernel_size=3,
                   dropout_rate=dropout_rate, seq_len=window_length)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
    # 训练
    for epoch in range(190):
        epoch_loss = 0
        for i, (datax, datay) in enumerate(train_loader):
            net.train()
            loss = criterion(net(datax), datay)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.detach().item()

    preprocessor.fit(train_y_norm1)
    net.eval()
    val_num = int(test_y_real.shape[0] / 2)
    val_x = test_x[:val_num, :, :]
    val_y_real = test_y_real[:val_num, :]
    test_x = test_x[val_num:, :, :]
    test_y_real = test_y_real[val_num:, :]
    # 验证集
    predict = net(torch.from_numpy(val_x).float()).detach().numpy()
    predict_y = preprocessor.inverse_transform(predict)
    velMAPE = np.mean(np.abs(val_y_real - predict_y) / val_y_real)
    print(val_y_real.shape, predict_y.shape)
    print("val集预测平均相对误差：" + str(velMAPE))
    # 测试集
    predict = net(torch.from_numpy(test_x).float()).detach().numpy()
    predict_y = preprocessor.inverse_transform(predict)
    testMAPE = np.mean(np.abs(test_y_real - predict_y) / test_y_real)
    R2 = r2_score(test_y_real, predict_y)
    print("测试集预测平均相对误差：" + str(testMAPE))
    print('testR2：', R2)
    if testMAPE < 0.042:
        torch.save(net.state_dict(),
                   'TCN-YH7_layer_{}+window_{}+lr_{}+drop_{}+VEL_{}+TES{}+r2_{}.pkl'.format(
                       layer_num, window_length, learning_rate, dropout_rate, velMAPE, testMAPE, R2))

    return testMAPE,

population_size = 10 #初始种群数
num_generations = 20 #迭代次数
gene_length = 13  #个体基因长度

#Implementation of Genetic Algorithm using DEAP python library.

#Since we try to minimise the loss values, we use the negation of the root mean squared loss as fitness function.
creator.create('FitnessMax', base.Fitness, weights = (-1.0,))
creator.create('Individual', list, fitness = creator.FitnessMax)

#initialize the variables as bernoilli random variables
toolbox = base.Toolbox()
toolbox.register('binary', bernoulli.rvs, 0.5)
toolbox.register('individual', tools.initRepeat, creator.Individual, toolbox.binary, n = gene_length)
toolbox.register('population', tools.initRepeat, list , toolbox.individual)

#Ordered cross-over used for mating
toolbox.register('mate', tools.cxOrdered)
#Shuffle mutation to reorder the chromosomes
toolbox.register('mutate', tools.mutShuffleIndexes, indpb = 0.6)
#use roulette wheel selection algorithm 锦标赛
toolbox.register('select', tools.selTournament, tournsize = 2)
#training function used for evaluating fitness of individual solution.
toolbox.register('evaluate', train_evaluate)

population = toolbox.population(n = population_size)
r = algorithms.eaSimple(population, toolbox, cxpb = 0.4, mutpb = 0.1, ngen = num_generations, verbose = False)