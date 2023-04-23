import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler

data = pd.read_csv("C:/Users/lolol/PycharmProjects/Выч мат № 4/Данные/Breast Cancer Wisconsin.csv", header=0)
diagnosis = np.array(data.loc[:, ["diagnosis"]])
only_data = np.array(data.iloc[:, 2:])
scale = StandardScaler()
scale.fit(only_data)
only_data = scale.transform(only_data)


def get_pca(Matr):
    W, V = np.linalg.eig(Matr.T @ Matr)
    ind = np.flip(np.argsort(np.sqrt(W)))
    sigma = np.sqrt(W)[ind]
    self_vec = V[:, ind]
    return self_vec.T, np.sqrt(1 / (Matr.shape[0] - 1)) * sigma


def get_normalized_data_matrix(Matr):
    m = Matr.shape[0]
    A = (np.eye(m) - 1. / m * np.ones((m, m))) @ Matr
    return A


scale = StandardScaler()
scale.fit(only_data)
only_data = scale.transform(only_data)

Matrix = get_normalized_data_matrix(only_data)
Q, Vector = get_pca(Matrix)

Q = Q[:2]
Matrix_k = Matrix @ Q.T

plt.subplots(figsize=(10, 6))
plt.plot(1. + np.arange(len(Vector)), Vector, "o--", color="grey")
plt.xlabel('index', fontsize=20)
plt.ylabel(r'Standard Deviation', fontsize=20)
plt.title("Standard Deviation From Component Number", fontsize=20)
plt.grid()
plt.show()

plt.subplots(figsize=(14, 6))
plt.xlabel('Principal Component - 1', fontsize=20)
plt.ylabel('Principal Component - 2', fontsize=20)
plt.title("Principal Component Analysis of Breast Cancer Dataset", fontsize=20)
for i in range(len(only_data)):
    if diagnosis[i][0] == "M":
        grey_dot = plt.scatter(Matrix_k[i][0], Matrix_k[i][1], color="grey")
    else:
        orange_dot = plt.scatter(Matrix_k[i][0], Matrix_k[i][1], color="orange")
plt.legend([grey_dot, orange_dot], ['Malignant', 'Benign'])
plt.grid()
plt.show()


def fiedler_vector(Matrix):
    w, v = np.linalg.eig(Matrix)
    f_i = np.argsort(w)[1]
    if len(Matrix) == 20:
        f_i = f_i + 1
    fiedler = []
    for i in range(len(v)):
        fiedler.append(v[i][f_i])
    return fiedler
