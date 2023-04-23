import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.colors import ListedColormap


def Laplas(Matrix):
    D = np.zeros((len(Matrix), len(Matrix)))
    for i in range(len(Matrix)):
        val = 0
        for j in range(len(Matrix)):
            if Matrix[i][j] == 1:
                val = val + 1
        D[i][i] = val
    return D - Matrix


def eig_values(L):
    w, v = np.linalg.eig(L)
    i = np.argsort(w)[1]
    # if len(L) == 10:
    #     print(np.array(w))
    #     print(np.array(v))
    return w,v, i


def Sort_matrix(v, ind, Matrix):
    v_stolb = []
    for i in range(len(v)):
        if len(v) == 20:
            v_stolb.append(v[i][ind + 1])
        else:
            v_stolb.append(v[i][ind])

    sort = pd.Series(v_stolb).sort_values(ascending=True)
    ind_s = sort.index.tolist()

    A = np.zeros((len(Matrix), len(Matrix)))
    for i in range(0, len(Matrix)):
        for j in range(0, len(Matrix)):
            A[i][j] = Matrix[ind_s[i]][ind_s[j]]

    return v_stolb, A


def graphs(v, A):
    v.sort()
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(1. + np.arange(len(v)), v, 'o', color="grey")
    plt.xlabel('index', fontsize=20)
    plt.ylabel(r'Vector values', fontsize=20)
    plt.title("Sorted Fiedler Vector", fontsize=20)
    plt.grid()
    plt.show()

    cmap = ListedColormap(['orange', 'grey'])
    plt.matshow(A, fignum=1, cmap=cmap)
    plt.title("Adjacencies Clustering")
    plt.show()


A1 = np.ones((10, 10))
A1[np.diag_indices(10)] = 0
L1 = Laplas(A1)
w1,v1, i1 = eig_values(L1)
v_stolb1, A1_sort = Sort_matrix(v1, i1, A1)
graphs(w1, A1)

A2 = np.zeros((20, 20))
A2[0][1] = A2[0][3] = A2[0][4] = 1
A2[1][2] = A2[1][7] = A2[1][4] = A2[1][0] = 1
A2[2][4] = A2[2][3] = A2[2][1] = 1
A2[3][2] = A2[3][0] = A2[3][6] = A2[3][19] = 1
A2[4][0] = A2[4][1] = A2[4][2] = A2[4][5] = A2[4][6] = A2[4][7] = 1
A2[5][4] = A2[5][7] = A2[5][6] = A2[5][12] = 1
A2[6][4] = A2[6][5] = A2[6][3] = A2[6][7] = A2[6][18] = 1
A2[7][1] = A2[7][4] = A2[7][5] = A2[7][6] = A2[7][14] = 1
A2[8][14] = A2[8][9] = 1
A2[9][8] = A2[9][10] = A2[9][11] = A2[9][13] = A2[9][14] = 1
A2[10][9] = A2[10][15] = A2[10][12] = A2[10][14] = 1
A2[11][14] = A2[11][12] = A2[11][9] = 1
A2[12][5] = A2[12][11] = A2[12][10] = A2[12][14] = A2[12][13] = 1
A2[13][12] = A2[13][14] = A2[13][9] = 1
A2[14][12] = A2[14][11] = A2[14][10] = A2[14][13] = A2[14][8] = A2[14][9] = A2[14][7] = 1
A2[15][10] = A2[15][16] = A2[15][17] = A2[15][18] = 1
A2[16][15] = A2[16][17] = A2[16][18] = A2[16][19] = 1
A2[17][15] = A2[17][16] = A2[17][18] = A2[17][19] = 1
A2[18][6] = A2[18][19] = A2[18][17] = A2[18][16] = A2[18][15] = 1
A2[19][3] = A2[19][18] = A2[19][16] = A2[19][17] = 1
L2 = Laplas(A2)
w2,v2, i2 = eig_values(L2)
v_stolb2, A2_sort = Sort_matrix(v2, i2, A2)
graphs(v_stolb2, A2_sort)

with open('../Данные/adjacency_matrix.txt') as f:
    A3 = [list(map(int, row.split())) for row in f.readlines()]
A3 = np.array(A3)
L3 = Laplas(A3)
w3,v3, i3 = eig_values(L3)
v_stolb3, A3_sort = Sort_matrix(v3, i3, A3)
graphs(w3, A3)
