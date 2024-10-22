import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
from sklearn.metrics import classification_report # 这个包是评价报告

import os

base_directory = r'C:\Users\admin\Desktop\ai\AI-study\吴恩达'   
data_directory = '逻辑回归\logistic_regression'
filename = 'ex2data1.txt'

file_path = os.path.join(base_directory, data_directory, filename)

# 检查文件是否存在
if os.path.exists(file_path):
    print("文件存在，开始读取数据。")
    data = pd.read_csv(file_path, header=None, names=['exam1', 'exam2', 'admitted'])
    print(data.head())
else:
    print("文件不存在。")

data.describe()

positive = data[data['admitted'].isin([1])]
negative = data[data['admitted'].isin([0])]

# 绘制散点图
# fig, ax = plt.subplots(figsize=(12,8))
# ax.scatter(positive['exam1'], positive['exam2'], s=50, c='b', marker='o', label='Admitted')
# ax.scatter(negative['exam1'], negative['exam2'], s=50, c='r', marker='x', label='Not Admitted')
# ax.legend()
# ax.set_xlabel('Exam 1 Score')
# ax.set_ylabel('Exam 2 Score')
# plt.show()

def get_X(df): # 读取特征

    ones = pd.DataFrame({'ones': np.ones(len(df))}) # ones是m行1列的dataframe, 值都是1
    data = pd.concat([ones, df], axis=1)  # 合并数据，根据列合并，合并在数据的左边
    return data.iloc[:, :-1].values  # 这个操作删除最后一列，返回 ndarray,不是矩阵


def get_y(df): # 读取标签
#     '''assume the last column is the target'''
    return np.array(df.iloc[:, -1])#df.iloc[:, -1]是指df的最后一列


def normalize_feature(df):
#     """Applies function along input axis(default 0) of DataFrame."""
    # lambda column:: 定义一个匿名函数，接收一个参数，参数名字是 column。在这个特定的例子中，column 就代表的是传入的 DataFrame 中的每一列。

    # (column - column.mean()) / column.std(): 这是函数体，它执行真正的操作。对于传入的每一列 column：

    # 计算每一列的均值 column.mean()。
    # 用列中的每个值减去均值进行中心化。
    # 将中心化后的值除以标准差 column.std()，达到缩放的目的。
    return df.apply(lambda column: (column - column.mean()) / column.std()) # 特征缩放

def sigmoid(z):
    # 逻辑函数
    return 1 / (1 + np.exp(-z))

# nums = np.arange(-10, 10, step=1)

# fig, ax = plt.subplots(figsize=(12,8))
# ax.plot(nums, sigmoid(nums), 'r')
# ax.set_xlabel('z', fontsize=18)
# ax.set_ylabel('g(z)', fontsize=18)
# ax.set_title('sigmoid function', fontsize=18)
# plt.show()

def cost(theta, X, y):
    # 代价函数
    return np.mean(-y * np.log(sigmoid(X @ theta)) - (1 - y) * np.log(1 - sigmoid(X @ theta)))

X = get_X(data)
y = get_y(data)
theta = np.zeros(3)

# 初始值计算代价函数，theta是0
cost(theta, X, y)

# Gradient的实现，利用的是循环的方法
def gradient(theta, X, y):
    # 梯度下降
    theta = np.matrix(theta)
    X = np.matrix(X)
    y = np.matrix(y).T
    
    parameters = int(theta.ravel().shape[1])
    grad = np.zeros(parameters)
    
    error = sigmoid(X @ theta.T) - y
    
    for i in range(parameters):
        term = X[:,i].T @ error
        grad[i] = term / len(X)
    
    return grad

# 另外一种实现的方法
def gradien2t(theta, X, y):
    return (1 / len(X)) * X.T @ (sigmoid(X @ theta) - y)

gradient(theta, X, y)

import scipy.optimize as opt
# 利用scipy中的优化函数寻找最优参数
res = opt.minimize(fun=cost, x0=theta, args=(X, y), method='Newton-CG', jac=gradient)
print(res)

# 优化后的参数
cost(res.x, X, y)

# 预测函数
def predict(theta, X):
    probability = sigmoid(X @ theta)
    return [1 if x >= 0.5 else 0 for x in probability]

# 测试下accuracy
theta_min = res.x
predictions = predict(theta_min, X)
correct = [1 if ((a == 1 and b == 1) or (a == 0 and b == 0)) else 0 for (a, b) in zip(predictions, y)]
accuracy = (sum(map(int, correct)) % len(correct))
print ('accuracy = {0}%'.format(accuracy))
print(classification_report(y, predictions))

# 寻找决策边界
# x0 = 1
# 定义x1的range后，根据 θ0 + x1*θ1 + x2*θ2 = 0，可得x2
x1 = np.arange(130, step=0.1)

coef = -(res.x / res.x[2])  # find the equation
x2 = coef[0] + coef[1]*x1

# 绘制决策边界
import seaborn as sns
sns.set(context="notebook", style="ticks", font_scale=1.5)

sns.lmplot(x='exam1', y='exam2', hue='admitted', data=data, 
           height=6, 
           fit_reg=False, 
           scatter_kws={"s": 25}
          )

plt.plot(x1, x2, 'grey')
plt.xlim(0, 130)
plt.ylim(0, 130)
plt.title('Decision Boundary')
plt.show()