import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

file = 'Data_Regression_11.csv'

z = pd.read_csv(file)
data = z.as_matrix()
m, n = data.shape
X = data[:, 0:n-1]
X = np.delete(X, 9, 1)

r, c = X.shape

Y = data[:, n-1]
Y = Y.reshape(-1, 1)
one = np.ones(shape=(m, 1))
X = np.concatenate((one, X), axis=1)
for i in range(1, c+1):
    X[:, i] = (X[:, i] - np.amin(X[:,i]))/(np.amax(X[:, i]) - np.amin(X[:, i]))
W = np.array([np.random.rand(c+1)])
W = W.T
X = np.matrix(X)
Y = np.matrix(Y)
W = np.matrix(W)
folds = 10;
F = KFold(n_splits=folds)
F.get_n_splits(X)
err = []
a = 0.001


for train, test in F.split(X):
    X_train, X_test = X[train], X[test]
    Y_train, Y_test = Y[train], Y[test]
    maxiter = 1000
    itr = 0
    while itr <= maxiter:
        y = X_train * W
        E = y - Y_train
        S = X_train.T * E
        W = W - (a * S)
        itr = itr+1
    y = X_test * W
    print(W)
    print(np.sum(np.array(y - Y_test) ** 2, axis=0))
    err.append(np.sum(np.array(y - Y_test) ** 2, axis=0))

print("average error :", np.sum(err, axis=0)/folds)

