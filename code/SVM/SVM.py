from sklearn.model_selection import GridSearchCV
from sklearn import preprocessing
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix,precision_score,recall_score
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn import svm
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

train_ratio = 0.7 #训练集比例1
forecast_horizon = 1  #预测步长
window_length = 3  #输入窗口步长
batch_size = 64
epoch = 300
learning_rate = 0.001
layer_num = 6
dropout_rate = 0.3

# df = pd.read_excel('../data/YH7data.xlsx')
# df = pd.read_excel('../data/JH126 - VMD.xlsx')
df = pd.read_excel('../data/YH7data.xlsx')
data_files = np.array(df)
data = data_files[:,1] #去掉时间一列
data = data.astype('float32')
# feature = data.shape[1]


#分割训练集和测试集
train_size = int(len(data) * 0.6) #train_ratio 7:3
print(train_size)
data = data.reshape([-1,1])
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

train_y_real = train_y_norm1[window_length:]
test_y_real = test_y_norm1[window_length:]

#训练集输入输出确定
train_x = windowed_dataset(train_x_norm[:-forecast_horizon], window_length).squeeze(axis=-1) #（num, seq, feature）
print(train_x.shape)
train_y = train_y_norm[window_length:].squeeze(axis=-1) #(169,1)
#测试集输入确定
test_x = windowed_dataset(test_x_norm[:-forecast_horizon], window_length).squeeze(axis=-1) #（34,12,13）

print(train_x.shape)
print(train_y.shape)

estimator = svm.SVR()

param_grid = {"C": [i for i in range(1,5)],
              "kernel": ['linear','rbf','poly']
              }

GS_model = GridSearchCV(estimator, param_grid, scoring='neg_mean_absolute_error', cv=10 ,verbose=1 ,n_jobs=-1)

GS_model.fit(train_x,train_y)
print('最优参数：',GS_model.best_params_)

net = GS_model.best_estimator_

preprocessor.fit(train_y_norm1)
predict_train = net.predict(train_x).reshape([-1,1])

predict_train = preprocessor.inverse_transform(predict_train).reshape([-1,1])
testMAPE = np.mean(np.abs(train_y_real - predict_train) / train_y_real)
print("训练集预测平均相对误差：" + str(testMAPE))
plt.figure()
plt.plot(train_y_real,  label='real')
plt.plot(predict_train, label='predict')
plt.legend()
# plt.show()

#输出验证集测试集预测数据
predict_all = net.predict(test_x).reshape([-1,1])
predict_all = preprocessor.inverse_transform(predict_all)

preprocessor.fit(train_y_norm1)
val_num = int(test_y_real.shape[0]/2)
val_x = test_x[:val_num,:]
val_y_real = test_y_real[:val_num,:]
test_x = test_x[val_num:,:]
test_y_real = test_y_real[val_num:,:]



#验证集
predict = net.predict(val_x).reshape([-1,1])
predict_y = preprocessor.inverse_transform(predict)
testMAPE = np.mean(np.abs(val_y_real - predict_y) / val_y_real)
MSE = mean_squared_error(val_y_real,predict_y)
RMSE = MSE ** 0.5
MAE = mean_absolute_error(val_y_real,predict_y)
R2 = r2_score(val_y_real,predict_y)
print(val_y_real.shape, predict_y.shape)
print("val集预测平均相对误差：" + str(testMAPE))
print('MSE: ' + str(MSE))
print('RMSE: ' + str(RMSE))
print('MAE: ' + str(MAE))
print('r2', R2)
plt.figure()
plt.plot(val_y_real,  label='val_real')
plt.plot(predict_y, label='val_predict')
plt.legend()
# plt.show()
#测试集
predict = net.predict(test_x).reshape([-1,1])
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

