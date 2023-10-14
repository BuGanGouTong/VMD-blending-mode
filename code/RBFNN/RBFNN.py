import torch, random
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import torch.utils.data as data_utils
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
torch.manual_seed(42)


class RBFN(nn.Module):
    def __init__(self, in_feature, center_num, n_out=1):
        super(RBFN, self).__init__()
        self.n_out = n_out
        self.n_in = in_feature
        self.num_centers = center_num #center = (center_num, input_dim)

        self.centers = nn.Parameter(torch.randn(center_num, in_feature))
        self.beta = nn.Parameter(torch.ones(1, self.num_centers), requires_grad=True) # beta = (1, num_center)
        # self.linear = nn.Linear(self.num_centers + self.n_in, self.n_out, bias=True)
        self.linear = nn.Linear(self.num_centers, self.n_out, bias=True)
        self.initialize_weights()

    def kernel_fun(self, batches):
        n_input = batches.size(0)  # number of inputs
        A = self.centers.view(self.num_centers, -1).repeat(n_input, 1, 1) # 高斯函数C = (n_input, num_center, input_dim)
        B = batches.view(n_input, -1).unsqueeze(1).repeat(1, self.num_centers, 1) # X = (n_input, input_dim) > (n_input, 1, input_dim) > (n_input, num_centers, input_dim)
        C = torch.exp(-self.beta.mul((A - B).pow(2).sum(2, keepdim=False))) # exp(- β * (x - c)^ 2)
        return C

    def forward(self, x):
        x = self.kernel_fun(x)
        # class_score = self.linear(torch.cat([batches, radial_val], dim=1))
        x = self.linear(x)
        return x

    def initialize_weights(self, ):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0, 0.02)
                m.bias.data.zero_()
            elif isinstance(m, nn.ConvTranspose2d):
                m.weight.data.normal_(0, 0.02)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.02)
                m.bias.data.zero_()

    def print_network(self):
        num_params = 0
        for param in self.parameters():
            num_params += param.numel()
        print(self)
        print('Total number of parameters: %d' % num_params)

class RBFN_TS(object):
    def __init__(self, args):
        self.max_epoch = args.epoch
        self.trainset = args.dataset[0]
        self.testset = args.dataset[1]
        self.model_name = args.model_name
        self.lr = args.lr
        self.n_in = args.n_in
        self.n_out = args.n_out
        self.num_centers = args.num_centers
        #  self.center_id = np.random.choice(len(self.trainset[0]),self.num_centers,replace=False)
        #  self.centers = torch.from_numpy(self.trainset[0][self.center_id]).float()
        self.centers = torch.rand(self.num_centers,self.n_in)

        self.model = RBFN(self.centers, n_out=self.n_out)

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.loss_fun = nn.MSELoss()

    def train(self, epoch=1):
        self.model.train()
        for epoch in range(min(epoch,self.max_epoch)):
            avg_cost = 0

            X = torch.from_numpy(self.trainset[0]).float()
            Y = torch.from_numpy(self.trainset[1]).float()        # label is not one-hot encoded

            self.optimizer.zero_grad()
            Y_prediction = self.model(X)
            cost = self.loss_fun(Y_prediction, Y)
            cost.backward()
            self.optimizer.step()

            print("[Epoch: {:>4}] cost = {:>.9}".format(epoch + 1, cost.item()))
        print(" [*] Training finished!")

    def test(self):
        self.model.eval()
        X = torch.from_numpy(self.testset[0]).float()
        Y = torch.from_numpy(self.testset[1]).float()        # label is not one-hot encoded

        with torch.no_grad():             # Zero Gradient Container
            Y_prediction = self.model(X)         # Forward Propagation
            cost = self.loss_fun(Y_prediction, Y[:,:3])

            print('Accuracy of the network on test data: %f' % cost.item())
            print(" [*] Testing finished!")

# class Dict(dict):
#     __setattr__ = dict.__setitem__
#     __getattr__ = dict.__getitem__
#
#
# args = Dict(
#     lr = 0.01,
#     epoch = 1000,
#     n_in = 3*n,
#     n_out = 3,
#     num_centers = N_h,  # 100
#     save_dir = 'ckpoints',
#     result_dir = 'outs',
#     dataset = [(x_train.T, y_train.T), (x_test.T, y_test.T)],
#     model_name='RBFN',
#     cuda=False
# )
#
# rbfn = RBFN_TS(args)
# rbfn.train(1000)
# rbfn.test()

df = pd.read_excel('模型预测数据/YH7-汇总-合并.xlsx')
# df = pd.read_excel('模型预测数据/汇总-126-合并.xlsx')
# df = pd.read_excel('模型预测数据/汇总-286-合并.xlsx')
data_files = np.array(df)
data = data_files[:,1:] #去掉时间一列
data = data.astype('float32')
d_input = data.shape[1]
data_x = data[:,0:-1]
data_y = data[:,-1:]

val_num = int(data.shape[0]/2)
val_x = data_x[:val_num,:]
val_y_real = data_y[:val_num,:]
test_x = data_x[val_num:,:]
test_y_real = data_y[val_num:,:]

def evaluationindex(predict, true):
    MAPE = np.mean(np.abs(true - predict) / true)
    MSE = mean_squared_error(true, predict)
    RMSE = MSE ** 0.5
    MAE = mean_absolute_error(true, predict)
    R2 = r2_score(true, predict)
    print("平均相对误差：" + str(MAPE))
    print('MSE: ' + str(MSE))
    print('RMSE: ' + str(RMSE))
    print('MAE: ' + str(MAE))
    print('r2', R2)
    return MAPE, MSE, RMSE, MAE, R2


# MAPE, MSE, RMSE, MAE, R2 = evaluationindex(test_x[:,0].reshape(-1,1), test_y_real)
#
# MAPE, MSE, RMSE, MAE, R2 = evaluationindex(test_x[:,1].reshape(-1,1), test_y_real)
#
# MAPE, MSE, RMSE, MAE, R2 = evaluationindex(test_x[:,2].reshape(-1,1), test_y_real)

# # 绘制预测值-真实值图
# fig, ax = plt.subplots()
# ax.scatter(test_x[:,0].reshape(-1,1), test_y_real)
# ax.plot([test_y_real.min(), test_y_real.max()], [test_y_real.min(), test_y_real.max()], ls="--", c=".3")
# ax.set_xlabel('GRU-Predicted Values')
# ax.set_ylabel('True Values')
# plt.show()
#
# # 绘制预测值-真实值图
# fig, ax = plt.subplots()
# ax.scatter(test_x[:,1].reshape(-1,1), test_y_real)
# ax.plot([test_y_real.min(), test_y_real.max()], [test_y_real.min(), test_y_real.max()], ls="--", c=".3")
# ax.set_xlabel('TCN-Predicted Values')
# ax.set_ylabel('True Values')
# plt.show()
#
# fig, ax = plt.subplots()
# ax.scatter(test_x[:,1].reshape(-1,1), test_y_real)
# ax.plot([test_y_real.min(), test_y_real.max()], [test_y_real.min(), test_y_real.max()], ls="--", c=".3")
# ax.set_xlabel('TF-Predicted Values')
# ax.set_ylabel('True Values')
# plt.show()


preprocessor = MinMaxScaler()
preprocessor.fit(data_x)
val_x_norm = preprocessor.transform(val_x)
test_x_norm = preprocessor.transform(test_x)

preprocessor.fit(data_y)
val_y_norm = preprocessor.transform(val_y_real)

val_data = data_utils.TensorDataset(torch.from_numpy(val_x_norm).float(), torch.from_numpy(val_y_norm).float())
val_loader = data_utils.DataLoader(val_data, batch_size=64, shuffle=False)



learning_rate  = 0.001

net = RBFN(in_feature=3, center_num=100, n_out=1)


# 训练
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate, weight_decay=0.0001)
epoch_losses = []
for epoch in range(1500):
    epoch_loss = 0
    for i, (datax, datay) in enumerate(val_loader):
        net.train()
        loss = criterion(net(datax), datay)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # scheduler.step()
        epoch_loss += loss.detach().item()
    epoch_loss /= (i + 1)
    # print('Epoch {}, loss {}'.format(epoch, epoch_loss))
    epoch_losses.append(epoch_loss)

# plt.figure()
# plt.plot(epoch_losses, 'b', label='loss')
# plt.xlabel('epoch')
# plt.ylabel('loss')
# plt.legend()
# net.eval()

preprocessor.fit(data_y)
predict_train = net(torch.from_numpy(val_x_norm).float()).detach().numpy()
predict_train = preprocessor.inverse_transform(predict_train)
testMAPE = np.mean(np.abs(val_y_real - predict_train) / val_y_real)
print("训练集预测平均相对误差：" + str(testMAPE))
# plt.figure()
# plt.plot(val_y_real,  label='real')
# plt.plot(predict_train, label='vel-predict')
# plt.legend()

#测试集
predict = net(torch.from_numpy(test_x_norm).float()).detach().numpy()
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

# plt.figure()
# plt.plot(test_y_real,  label='real')
# plt.plot(predict_y, label='test-predict')
# plt.legend()
# plt.show()


# 绘制预测值-真实值图
fig, ax = plt.subplots()
# ax.scatter(test_x[:,0].reshape(-1,1), test_y_real, s=15, label='VMD-GRU')
# ax.scatter(test_x[:,1].reshape(-1,1), test_y_real, s=15, label='VMD-TCN')
# ax.scatter(test_x[:,1].reshape(-1,1), test_y_real, s=15, label='VMD-Transformer')

ax.scatter(val_y_real, predict_train, s=10, label='Training Set')
ax.scatter(test_y_real, predict_y, c='r', s=10, label='Validation Set')
ax.plot([data_y.min(), data_y.max()], [data_y.min(), data_y.max()], ls="--", c=".3")
ax.set_xlabel('True Values')
ax.set_ylabel('Predicted Values')
ax.legend( )
plt.show()


