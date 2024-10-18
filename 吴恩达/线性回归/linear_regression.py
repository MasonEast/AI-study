import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

import os

sns.set(context="notebook", style="whitegrid", palette="dark")

base_directory = r'C:\Users\admin\Desktop\ai\AI-study\吴恩达'   
data_directory = '线性回归'
filename = 'ex1data1.txt'

file_path = os.path.join(base_directory, data_directory, filename)

# 检查文件是否存在
if os.path.exists(file_path):
    print("文件存在，开始读取数据。")
    df = pd.read_csv(file_path, names=['population', 'profit'])
    print(df.head())
else:
    print("文件不存在。")

df.head() # 显示数据前五行
df.info() # 打印df的class信息
df.describe() # 打印df的统计信息

# 看下原始数据
sns.lmplot(x='population', y='profit', data=df, height=6, aspect=1, fit_reg=False)
plt.show()

def get_X(df): # 读取特征

    ones = pd.DataFrame({'ones': np.ones(len(df))})#ones是m行1列的dataframe
    data = pd.concat([ones, df], axis=1)  # 合并数据，根据列合并
    return data.iloc[:, :-1].values  # 这个操作返回 ndarray,不是矩阵


def get_y(df):#读取标签
#     '''assume the last column is the target'''
    return np.array(df.iloc[:, -1])#df.iloc[:, -1]是指df的最后一列


def normalize_feature(df):
#     """Applies function along input axis(default 0) of DataFrame."""
    return df.apply(lambda column: (column - column.mean()) / column.std())#特征缩放

# 查看数据维度
data = df
X = get_X(data)
print(X.shape, type(X))

y = get_y(data)
print(y.shape, type(y))

theta = np.zeros(X.shape[1]) # X.shape[1]=2, 代表特征数n
print(theta)

def lr_cost(theta, X, y):
    """ 计算代价函数
    X: R(m*n), m 样本数, n 特征数
    y: R(m)
    theta : R(n), 线性回归的参数
    """
    m = X.shape[0]#m为样本数

    inner = X @ theta - y  # R(m*1)，X @ theta等价于X.dot(theta)

    square_sum = inner.T @ inner
    cost = square_sum / (2 * m)

    return cost

lr_cost(theta, X, y) # 返回cost的值

def gradient(theta, X, y):
  """
  计算梯度，也就是 J(θ)的偏导数
  """
  m = X.shape[0]

  inner = X.T @ (X @ theta - y)  # (m,n).T @ (m, 1) -> (n, 1)，X @ theta等价于X.dot(theta)

  return inner / m

def batch_gradient_decent(theta, X, y, epoch, alpha=0.01):
  """
  批量梯度下降函数。拟合线性回归，返回参数和代价
    epoch: 批处理的轮数
  """
  cost_data = [lr_cost(theta, X, y)]
  _theta = theta.copy()  # 拷贝一份，不和原来的theta混淆

  for _ in range(epoch):
    _theta = _theta - alpha * gradient(_theta, X, y)
    cost_data.append(lr_cost(_theta, X, y))

  return _theta, cost_data

epoch = 500
final_theta, cost_data = batch_gradient_decent(theta, X, y, epoch)

final_theta

cost_data

# 计算最终的代价
lr_cost(final_theta, X, y)

from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(X, y)

x = X[:, 1]
f = model.predict(X).flatten()

plt.scatter(X[:,1], y, label='Traning Data')
plt.plot(x, f, 'r', label='Prediction')
plt.legend(loc=2)
plt.show()

# 使用 lineplot
sns.set(style="whitegrid")
ax = sns.lineplot(x=np.arange(epoch+1), y=cost_data)

plt.xlabel("Epoch")
plt.ylabel("Cost")
plt.title("Cost Data over Epochs")
plt.show()
#可以看到从第二轮代价数据变换很大，接下来平稳了

b = final_theta[0] # intercept，Y轴上的截距
m = final_theta[1] # slope，斜率

plt.scatter(data.population, data.profit, label="Training data")
plt.plot(data.population, data.population*m + b, 'r', label="Prediction")
plt.legend(loc=2)
plt.show()