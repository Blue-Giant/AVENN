import tensorflow as tf
import numpy as np
# 偏微分方程的一些信息:边界条件，初始条件，真解，右端项函数


def construct_matrix1(row, cloumn):
    #  A = tridiag(-1,8,-1)
    row = int(row)
    cloumn = int(cloumn)
    M = np.eye(row, dtype=float)*8 - np.eye(row, k=1) - np.eye(row, k=-1)
    M = M.astype(np.float32)
    return M


def construct_matrix2(row, cloumn):
    # A = tridiag(-I,S,-I), S = = tridiag(-1,8,-1), I=eye(row)
    row = int(row)
    I = np.eye(row)
    I = I.astype(np.float32)
    M = np.eye(row, dtype=float) * 4 - np.eye(row, k=1) - np.eye(row, k=-1)
    M = M.astype(np.float32)
    mat = np.kron(I, M) + np.kron(M, I)
    return mat


def construct_matrix3(row, cloumn):
    row = int(row)
    cloumn = int(cloumn)
    M = np.eye(row, dtype=float)*16 - 4*np.eye(row, k=1) - 4*np.eye(row, k=-1) - 2*np.eye(row, k=2) - \
        2*np.eye(row, k=-2) - np.eye(row, k=row-2) - np.eye(row, k=-row+2) - np.eye(row, k=row-1) - np.eye(row, k=-row+1)
    M = M.astype(np.float32)
    return M


def construct_matrix4(row, cloumn):
    row = int(row)
    cloumn = int(cloumn)
    M = np.eye(row, dtype=float)*16 - 4*np.eye(row, k=1) - 4*np.eye(row, k=-1) - 2*np.eye(row, k=2) - \
        2*np.eye(row, k=-2) - np.eye(row, k=row-2) - np.eye(row, k=-row+2) - np.eye(row, k=row-1) - np.eye(row, k=-row+1)
    M = M.astype(np.float32)
    return M


def construct_matrix5(row, cloumn):
    row = int(row)
    cloumn = int(cloumn)
    M = np.eye(row, dtype=float)*16 - 4*np.eye(row, k=2) - 4*np.eye(row, k=-2) - 2*np.eye(row, k=4) - \
        2*np.eye(row, k=-4) - np.eye(row, k=row-3) - np.eye(row, k=-row+3) - np.eye(row, k=row-2) - \
        np.eye(row, k=-row+2) - np.eye(row, k=row-1) - np.eye(row, k=-row+1)
    M = M.astype(np.float32)
    return M


def construct_matrix6(row, cloumn):
    row = int(row)
    cloumn = int(cloumn)
    M = np.eye(row, dtype=float)*13 + 3*np.eye(row, k=2) - 4*np.eye(row, k=-2) - 2*np.eye(row, k=4) + \
        2*np.eye(row, k=-4) - np.eye(row, k=row-3) + np.eye(row, k=-row+3) + np.eye(row, k=row-2) - \
        np.eye(row, k=-row+2) - np.eye(row, k=row-1) + np.eye(row, k=-row+1)
    M = M.astype(np.float32)
    return M


def construct_matrix7(row, cloumn, mu=-5.0):
    # A = tridiag(-I,S,-I), S = = tridiag(-1,8,-1), I=eye(row)
    row = int(row)
    I = np.eye(row)
    I = I.astype(np.float32)
    M = np.eye(row, dtype=float) * 4 - np.eye(row, k=1) - np.eye(row, k=-1)
    M = M.astype(np.float32)
    mat = np.kron(I, M) + np.kron(M, I)
    Mat_add = mu * np.eye(row*row, dtype=float)
    Mat_add = Mat_add.astype(np.float32)
    matA = mat + Mat_add
    return matA


def construct_Hilbert_matrix(row, cloumn):
    row = int(row)
    cloumn = int(cloumn)
    M = np.zeros(shape=(row, cloumn))
    for i in range(1, row+1):
        for j in range(1, cloumn+1):
            M[i-1, j-1] = 1.0/(i+j-1)
    M = M.astype(np.float32)
    return M


def get_AVE_infos(Row=4, Column=4, AVE_equa_name=None):
    if AVE_equa_name == 'matrix1':
        Matrix_A = construct_matrix1(Row, Column)
        # Matrix_B = construct_matrix(Row, Column)
        Matrix_B = np.eye(Row)
        Matrix_B = Matrix_B.astype(np.float32)

        u_true = np.ones(shape=(Row, 1), dtype=np.float32)
        for i in range(0, Row):
            u_true[i, 0] = np.power(-1, i + 1) * u_true[i, 0]
            # u_true[0, i] = np.power(-1, i + 1) * u_true[0, i] * 5
    elif AVE_equa_name == 'matrix2':
        Matrix_A = construct_matrix2(Row, Column)
        # Matrix_B = construct_matrix(Row, Column)
        Matrix_B = np.eye(Row*Row)
        Matrix_B = Matrix_B.astype(np.float32)
        u_true = np.ones(shape=(Row * Row, 1), dtype=np.float32)
        for i in range(0, Row * Row):
            u_true[i, 0] = np.power(-1, i + 1) * u_true[i, 0]
            # u_true[0, i] = np.power(-1, i + 1) * u_true[0, i] * 5
    elif AVE_equa_name == 'matrix3':
        Matrix_A = construct_matrix3(Row, Column)
        # Matrix_B = construct_matrix(Row, Column)
        Matrix_B = np.eye(Row)
        Matrix_B = Matrix_B.astype(np.float32)

        u_true = np.ones(shape=(Row, 1), dtype=np.float32)
        for i in range(0, Row):
            u_true[i, 0] = np.power(-1, i + 1) * u_true[i, 0]
            # u_true[0, i] = np.power(-1, i + 1) * u_true[0, i] * 5
    elif AVE_equa_name == 'matrix4':
        Matrix_A = construct_matrix4(Row, Column)
        # Matrix_B = construct_matrix(Row, Column)
        Matrix_B = np.eye(Row)
        Matrix_B = Matrix_B.astype(np.float32)

        u_true = np.ones(shape=(Row, 1), dtype=np.float32)
        for i in range(0, Row):
            u_true[i, 0] = np.power(-1, i + 1) * u_true[i, 0]
            # u_true[0, i] = np.power(-1, i + 1) * u_true[0, i] * 5
    elif AVE_equa_name == 'matrix5':
        Matrix_A = construct_matrix5(Row, Column)
        # Matrix_B = construct_matrix(Row, Column)
        Matrix_B = np.eye(Row)
        Matrix_B = Matrix_B.astype(np.float32)

        u_true = np.ones(shape=(Row, 1), dtype=np.float32)
        for i in range(0, Row):
            u_true[i, 0] = np.power(-1, i + 1) * u_true[i, 0]
            # u_true[0, i] = np.power(-1, i + 1) * u_true[0, i] * 5
    elif AVE_equa_name == 'matrix6':
        Matrix_A = construct_matrix6(Row, Column)
        Matrix_B = np.eye(Row)
        Matrix_B = Matrix_B.astype(np.float32)

        u_true = np.ones(shape=(Row, 1), dtype=np.float32)
        for i in range(0, Row):
            u_true[i, 0] = np.power(-1, i + 1) * u_true[i, 0]
    elif AVE_equa_name == 'matrix7':
        Matrix_A = construct_matrix7(Row, Column)
        # Matrix_B = construct_matrix(Row, Column)
        Matrix_B = np.eye(Row*Row)
        Matrix_B = Matrix_B.astype(np.float32)
        u_true = np.ones(shape=(Row * Row, 1), dtype=np.float32)
        for i in range(0, Row * Row):
            u_true[i, 0] = np.power(-1, i + 1) * u_true[i, 0]
            # u_true[0, i] = np.power(-1, i + 1) * u_true[0, i] * 5
    elif AVE_equa_name == 'Hilbert':
        Matrix_A = construct_Hilbert_matrix(Row, Column)
        # Matrix_B = construct_matrix(Row, Column)
        Matrix_B = np.eye(Row)
        Matrix_B = Matrix_B.astype(np.float32)

        u_true = np.ones(shape=(Row, 1), dtype=np.float32)
        for i in range(0, Row):
            u_true[i, 0] = np.power(-1, i + 1) * u_true[i, 0]
            # u_true[0, i] = np.power(-1, i + 1) * u_true[0, i] * 5

    force_side = tf.matmul(Matrix_A, u_true) - tf.matmul(Matrix_B, np.abs(u_true))
    return Matrix_A, Matrix_B, u_true, force_side


def get_AVE_infos2no_square(Row=4, Column=4, AVE_equa_name=None):
    if AVE_equa_name == 'matrix0':
        row = int(Row)
        cloumn = int(Column)
        M = np.eye(row, dtype=float) * 8 - np.eye(row, k=1) - np.eye(row, k=-1)
        M = M.astype(np.float32)
        Matrix_A = M[:, 0:cloumn]
        Matrix_B = M[:, 0:cloumn]
        u_true = np.ones(shape=(Column, 1), dtype=np.float32)
        for i in range(0, Column):
            u_true[i, 0] = np.power(-1, i + 1) * u_true[i, 0]

        b_side = tf.matmul(Matrix_A, u_true) - tf.matmul(Matrix_B, np.abs(u_true))
        return Matrix_A, Matrix_B, u_true, b_side
    if AVE_equa_name == 'matrix1':
        row = int(Row)
        cloumn = int(row)
        Matrix_A = np.eye(row, dtype=float) * 5 + 2*np.ones(row, dtype=float)
        Matrix_B = np.eye(row, dtype=float) * 3 - np.eye(row, k=1) - np.eye(row, k=-1)
        Matrix_A = Matrix_A.astype(np.float32)
        Matrix_B = Matrix_B.astype(np.float32)

        u_true = np.ones(shape=(cloumn, 1), dtype=np.float32)
        for i in range(0, cloumn):
            u_true[i, 0] = np.power(-1, i + 1) * u_true[i, 0]

        b_side = np.matmul(Matrix_A, u_true) - np.matmul(Matrix_B, np.abs(u_true))
        return Matrix_A, Matrix_B, u_true, b_side
