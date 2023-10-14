import torch
import pandas as pd
import numpy as np
import torch.utils.data as data_utils
from torch import nn
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


class GRU(nn.Module):
    '''
    input_shape = (N, seq, fearure)
    output_shape = (N, seq, filter_num)
    '''
    def __init__(self, layer_num, n_input, n_filter, dropout_rate, seq_len):
        super(GRU, self).__init__()
        self.gru = nn.GRU(n_input, n_filter, layer_num, batch_first=True, dropout=dropout_rate)
        self.fc1 = nn.Linear(n_filter, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x, h_0 = self.gru(x)
        x = self.fc1(x[:, -1, :])

        return x

# data = torch.randn(50, 7, 10)
# net = GRU(layer_num=3, n_input=10, n_filter=64, kernel_size=3, dropout_rate=0.1, seq_len=7)
# print(net(data).shape)

train_ratio = 0.6 #训练集比例1
forecast_horizon = 1  #预测步长
window_length = 8  #输入窗口步长
batch_size = 64
epoch = 150
learning_rate = 0.0001
layer_num = 4
dropout_rate = 0.3

# df = pd.read_excel('../data/YH5data.xlsx')
# df = pd.read_excel('../data/JH126.xlsx')
# df = pd.read_excel('../data/JH126 - VMD.xlsx')
df = pd.read_excel('../data/JH008 - VMD.xlsx')
# df = pd.read_excel('../data/JH067 - VMD.xlsx')
data_files = np.array(df)
data = data_files[:,1:] #去掉时间一列
data = data.astype('float32')
feature = data.shape[1]
net = GRU(layer_num=layer_num, n_input=feature, n_filter=128, dropout_rate=dropout_rate, seq_len=window_length)
#分割训练集和测试集
train_size = int(len(data) * train_ratio) #train_ratio
data_train = data[0:train_size,:]
data_test = data[train_size:,:]

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
    return shape = (N, feature_num, seq_len)
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
train_x = windowed_dataset(train_x_norm[:-forecast_horizon], window_length) #（num, seq, feature）

train_y = train_y_norm[window_length:] #(169,1)
#测试集输入确定
test_x = windowed_dataset(test_x_norm[:-forecast_horizon], window_length) #（34,12,13）

#装载训练数据
train_data = data_utils.TensorDataset(torch.from_numpy(train_x).float(), torch.from_numpy(train_y).float())
train_loader = data_utils.DataLoader(train_data, batch_size=batch_size, shuffle=False)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate, weight_decay=0.0001)
# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.9)

# 训练
epoch_losses = []
for epoch in range(epoch):
    epoch_loss = 0
    for i, (datax, datay) in enumerate(train_loader):
        net.train()
        loss = criterion(net(datax), datay)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # scheduler.step()
        epoch_loss += loss.detach().item()
    epoch_loss /= (i + 1)
    print('Epoch {}, loss {}'.format(epoch, epoch_loss))
    epoch_losses.append(epoch_loss)

plt.figure()
plt.plot(epoch_losses, 'b', label='loss')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend()
# plt.show()
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

# if testMAPE < 0.05:
#     torch.save(net.state_dict(),
#                'GRU_window_{}+LR_{}+layer_{}+drop_{}+MAPE_{}.pkl'.format(
#                window_length, learning_rate, layer_num, dropout_rate, testMAPE))
