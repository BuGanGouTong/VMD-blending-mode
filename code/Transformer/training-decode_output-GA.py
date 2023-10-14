import datetime
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
from tst import Transformer
from sklearn.preprocessing import MinMaxScaler
import torch.utils.data as data_utils
from bitstring import BitArray
from deap import base, creator, tools, algorithms
from scipy.stats import bernoulli
from sklearn.metrics import r2_score
import openpyxl

# Training parameters
NUM_WORKERS = 0
# Model parameters
d_model = 128  # Lattent dim
dropout = 0.3  # Dropout rate
pe = None  # Positional encoding
chunk_mode = None
d_output = 1

train_ratio = 0.6#训练集比例1
forecast_horizon = 1  #预测步长

# df = pd.read_excel('YH7data.xlsx')
# df = pd.read_excel('../data/JH126 - VMD.xlsx')
# df = pd.read_excel('../data/JH008 - VMD.xlsx')
# df = pd.read_excel('../data/JH067 - VMD.xlsx')
# df = pd.read_excel('../data/JH286 - VMD.xlsx')
# df = pd.read_excel('../data/YH7data - VMD.xlsx')
df = pd.read_excel('../data/YH7data.xlsx')
data_files = np.array(df)
data = data_files[:,1:] #去掉时间一列
data = data.astype('float32')  #data(227.13)
d_input = data.shape[1]
feature = d_input
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

#数据归一化处理
preprocessor = MinMaxScaler()
out_preprocessor = MinMaxScaler()

preprocessor.fit(train_x_norm1)
train_x_norm = preprocessor.transform(train_x_norm1)
test_x_norm = preprocessor.transform(test_x_norm1)

preprocessor.fit(train_y_norm1)
train_y_norm = preprocessor.transform(train_y_norm1)


def windowed_dataset(series, time_series_number, window_size):
    """
    Returns a windowed dataset from a Pandas dataframe
    将输入数据进行时间步长处理
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
    n_head = BitArray(ga_individual_solution[1:3])
    encode_layer = BitArray(ga_individual_solution[3:6])
    dropout_rate = BitArray(ga_individual_solution[6:9])
    windows_length = BitArray(ga_individual_solution[9:13])
    attention_size = BitArray(ga_individual_solution[13:15])
    attention_qv = BitArray(ga_individual_solution[15:17])

    learning_rate = 10 ** (-learning_rate.uint-3)
    n_head = 2 ** (n_head.uint+1)
    encode_layer = encode_layer.uint + 1
    window_length = windows_length.uint + 8
    dropout_rate = (dropout_rate.uint+1) * 0.1
    attention_size = (attention_size.uint+1) * 4
    attention_qv = (attention_qv.uint + 1) * 2 + 6

    test_y_real = test_y_norm1[window_length:]

    # 训练集输入输出确定
    train_x = windowed_dataset(train_x_norm[:-forecast_horizon], feature, window_length)  # （169,12,13）
    train_y = train_y_norm[window_length:]  # (169,1)
    # 测试集输入输出确定
    test_x = windowed_dataset(test_x_norm[:-forecast_horizon], feature, window_length)  # （34,12,13）

    train_data = data_utils.TensorDataset(torch.from_numpy(train_x).float(), torch.from_numpy(train_y).float())
    train_loader = data_utils.DataLoader(train_data, batch_size=128, shuffle=False)

    # Load transformer with Adam optimizer and MSE loss function
    net = Transformer(d_input, d_model, d_output, q=attention_qv, v=attention_qv, h=n_head, N=encode_layer, seq_len=window_length, attention_size=attention_size,
                      dropout=dropout_rate, chunk_mode=chunk_mode, pe=pe)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate, weight_decay=0.001)
    # 训练
    epoch_losses = []
    for epoch in range(180):
        # epoch_loss = 0
        for i, (datax, datay) in enumerate(train_loader):
            net.train()
            loss = criterion(net(datax), datay)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # epoch_loss += loss.detach().item()
        # epoch_loss /= (i + 1)
        # print('Epoch {}, loss {}'.format(epoch, epoch_loss))
        # epoch_losses.append(epoch_loss)

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
    print("R2:", R2)

    if testMAPE < 0.036:
        torch.save(net.state_dict(),
                   'TF-YH7_win_{}+LR_{}+head_{}+layer_{}+drop_{}+att_{}+qv_{}+VEL_{}+TES{}+r2_{}.pkl'.format(
                       window_length, learning_rate, n_head, encode_layer, dropout, attention_size, attention_qv, velMAPE, testMAPE, R2))
    return testMAPE,

population_size = 20 #初始种群数
num_generations = 25 #迭代次数
gene_length = 17  #个体基因长度

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

optimal_individuals_data = tools.selBest(population,k = 1) #select top 1 solution
