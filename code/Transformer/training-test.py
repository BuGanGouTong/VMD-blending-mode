import datetime
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
from tst import Transformer
from sklearn.preprocessing import MinMaxScaler
import torch.utils.data as data_utils
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# Training parameters
BATCH_SIZE = 8
LR = 2e-4

# Model parameters
d_model = 256  # Lattent dim
attentionqv = 12
q = attentionqv  # Query size
v = attentionqv  # Value size
h = 4  # Number of heads
N = 2  # Number of encoder and decoder to stack
attention_size = 12  # Attention window size
dropout = 0.2  # Dropout rate
pe = None  # Positional encoding
chunk_mode = None

d_output = 1  # From dataset
train_ratio = 0.6 #训练集比例1
forecast_horizon = 1  #预测步长
window_length = 9  #输入窗口步长
batch_size = 64
epoch = 180
learning_rate = 0.0001

# Config
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device {device}")


# df = pd.read_excel('YH7data.xlsx')
df = pd.read_excel('../data/JH286 - VMD.xlsx')
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


train_y_real = train_y_norm1[window_length:]
test_y_real = test_y_norm1[window_length:]

#训练集输入输出确定
train_x = windowed_dataset(train_x_norm[:-forecast_horizon], feature, window_length) #（169,12,13）
train_y = train_y_norm[window_length:] #(169,1)
#测试集输入输出确定
test_x = windowed_dataset(test_x_norm[:-forecast_horizon], feature, window_length) #（34,12,13）

train_data = data_utils.TensorDataset(torch.from_numpy(train_x).float(), torch.from_numpy(train_y).float())
# test_data = data_utils.TensorDataset(torch.from_numpy(test_x).float(), torch.from_numpy(test_y).float())

train_loader = data_utils.DataLoader(train_data, batch_size=batch_size, shuffle=False)

net = Transformer(d_input, d_model, d_output, q, v, h, N,seq_len=window_length, attention_size=attention_size,
                  dropout=dropout, chunk_mode=chunk_mode, pe=pe).to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
#训练
epoch_losses = []
for epoch in range(epoch):
    epoch_loss = 0
    for i, (datax, datay) in enumerate(train_loader):
        net.train()
        loss = criterion(net(datax), datay)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.detach().item()
    epoch_loss /= (i + 1)
    print('Epoch {}, loss {}'.format(epoch, epoch_loss))
    epoch_losses.append(epoch_loss)

plt.figure()
plt.plot(epoch_losses, 'b', label='loss')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend()

net.eval()
preprocessor.fit(train_y_norm1)
predict_train = net(torch.from_numpy(train_x).float()).detach().numpy()
predict_train = preprocessor.inverse_transform(predict_train)
testMAPE = np.mean(np.abs(train_y_real - predict_train) / train_y_real)
print("训练集预测平均相对误差：" + str(testMAPE))
plt.figure()
plt.plot(train_y_real,  label='real')
plt.plot(predict_train, label='predict')
plt.legend()
# plt.show()

preprocessor.fit(train_y_norm1)
val_num = int(test_y_real.shape[0]/2)
val_x = test_x[:val_num,:,:]
val_y_real = test_y_real[:val_num,:]
test_x = test_x[val_num:,:,:]
test_y_real = test_y_real[val_num:,:]
#验证集
predict = net(torch.from_numpy(val_x).float()).detach().numpy()
predict_y = preprocessor.inverse_transform(predict)
velMAPE = np.mean(np.abs(val_y_real - predict_y) / val_y_real)
MSE = mean_squared_error(val_y_real,predict_y)
RMSE = MSE ** 0.5
MAE = mean_absolute_error(val_y_real,predict_y)
R2 = r2_score(val_y_real,predict_y)
print(val_y_real.shape, predict_y.shape)
print("val集预测平均相对误差：" + str(velMAPE))
print('MSE: ' + str(MSE))
print('RMSE: ' + str(RMSE))
print('MAE: ' + str(MAE))
print('r2', R2)
plt.figure()
plt.plot(val_y_real,  label='val_real')
plt.plot(predict_y, label='test_predict')
plt.legend()
# plt.show()
#测试集
predict = net(torch.from_numpy(test_x).float()).detach().numpy()
predict_y = preprocessor.inverse_transform(predict)
testMAPE = np.mean(np.abs(test_y_real - predict_y) / test_y_real)
MSE = mean_squared_error(test_y_real,predict_y)
RMSE = MSE ** 0.5
MAE = mean_absolute_error(test_y_real,predict_y)
R2 = r2_score(test_y_real,predict_y)
print(test_y_real.shape, predict_y.shape)
print("test集预测平均相对误差：" + str(testMAPE))
print('MSE: ' + str(MSE))
print('RMSE: ' + str(RMSE))
print('MAE: ' + str(MAE))
print('r2', R2)
plt.figure()
plt.plot(test_y_real,  label='test_real')
plt.plot(predict_y, label='test_predict')
plt.legend()

# 绘制预测值-真实值图
fig, ax = plt.subplots()
ax.scatter(predict_y, test_y_real)
ax.plot([test_y_real.min(), test_y_real.max()], [test_y_real.min(), test_y_real.max()], ls="--", c=".3")
ax.set_xlabel('Predicted Values')
ax.set_ylabel('True Values')
plt.show()