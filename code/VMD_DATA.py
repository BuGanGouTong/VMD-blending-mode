from vmdpy import VMD
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#读取数据

# df = pd.read_excel('../data/JH126.xlsx')
# df = pd.read_excel('../data/JH008.xlsx')
# df = pd.read_excel('JH286.xlsx')
df = pd.read_excel('YH5data.xlsx')
df1 = pd.read_excel('YH7data.xlsx')
data_files1 = np.array(df1)
data1 = data_files1[:,1]
data1 = data1.astype('float32')

data_files = np.array(df)
data = data_files[:,1]
data = data.astype('float32')

alpha = 2000      # moderate bandwidth constraint
tau = 0.            # noise-tolerance (no strict fidelity enforcement)
K = 5            # 3 modes
DC = 0             # no DC part imposed
init = 1           # initialize omegas uniformly
tol = 1e-7

u, u_hat, omega = VMD(data, alpha, tau, K, DC, init, tol)
plt.figure()

print(u.shape)
plt.plot(u.T)
plt.title('Decomposed modes')

df = pd.DataFrame(u.T)
print(df)
df.to_excel('VMDdata.xlsx')
u1, u_hat, omega = VMD(data1, alpha, tau, K, DC, init, tol)
fig1 = plt.figure()
plt.plot(data)
fig1.suptitle('Original input signal and its components')
fig2 = plt.figure(figsize=(10, 8))
for i in range(K):
    plt.subplot(K+1, 2, 1)
    plt.plot(data,c='r')
    plt.title("Well A")
    plt.text(8,30 , 'Original',style='normal',fontsize=10)
    # plt.suptitle("Original",x=0.1, y=0.9)
    plt.subplot(K+1, 2, i*2+3)
    plt.plot(u[i, :], linewidth=0.2, c='r')
    plt.title('u{}'.format(i + 1),x=0.1, y=0.65)
for i in range(K):
    plt.subplot(K+1, 2, 2)
    plt.plot(data1,c='c')
    plt.title("Well B")
    plt.text(8, 18, 'Original', style='normal', fontsize=10)
    # plt.suptitle("Original1", x=0.5, y=0.9)
    plt.subplot(K+1, 2, i*2+4)
    plt.plot(u1[i, :], linewidth=0.2, c='c')
    plt.title('u{}'.format(i + 1),x=0.1, y=0.65)
plt.tight_layout()
plt.show()
