import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd
from scipy import io
from sklearn import preprocessing
def sigmoid(z):
    s=1/(1+np.exp(-z))
    return s
def cost(theta, X, y):
    theta = np.matrix(theta)
    num=len(y)
    X = np.matrix(X)
    y = np.matrix(y)
    z=sigmoid(np.dot(X,theta.T))
    sum=-(1/num*np.sum(np.multiply(np.log(z),y)+np.multiply(np.log(1-z),1-y)))
    return sum


def gradient(theta, X, y):
    theta = np.matrix(theta)
    X = np.matrix(X)
    y = np.matrix(y)

    parameters = int(theta.ravel().shape[1])
    grad = np.zeros(parameters)
    m = len(y)

    error = sigmoid(X * theta.T) - y
    #error = 1/m*(sigmoid(X * theta.T) - y)    矩阵做法
    #grad =np.dot(error.T, X)

    for i in range(parameters):
    # 请将下述代码补全，该部分代码用于计算各个特征的梯度，梯度值存储在变量grad中
        grad[i]=1/m*X.T[i]*error


    return grad


path = 'ex2data1.txt'
data = pd.read_csv(path, header=None, names=['Exam 1', 'Exam 2', 'Admitted'])
data.head()
data.insert(0, 'Ones', 1)
# add a ones column - this makes the matrix multiplication work out easier

# set X (training data) and y (target variable)
cols = data.shape[1]
X = data.iloc[:,0:cols-1]
y = data.iloc[:,cols-1:cols]

# convert to numpy arrays and initalize the parameter array theta
X = np.array(X.values)
y = np.array(y.values)
theta = np.zeros(3)

# set X (training data) and y (target variable)


# convert to numpy arrays and initalize the parameter array theta

huafe=cost(theta, X, y)
print("cost:",huafe)
print(gradient(theta, X, y))