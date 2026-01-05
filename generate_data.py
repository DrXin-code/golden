import numpy as np
from scipy.stats import ortho_group
import torch

# from save2csv import save_data_to_path, save_matrix2csv
# ##log1-20###
# features_num = 20
# def generate_matrix_a():
#     W = ([1.5, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 1.51, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 1.52, 0, 0, 0, 0, 0, 0, 0],
#          [0, 0, 0, 1.51, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 1.53, 0, 0, 0, 0, 0])
#     W = np.hstack((np.array(W), np.zeros((5, features_num - 10))))  #从W的5维扩充到20维
#     V = ortho_group.rvs(20)
#     A = np.dot(W, V)
#     return A
#
# def get_data(random_seed=123, data_times=1, features_num=20):
#     np.random.seed(random_seed)
#     x = np.random.normal(0, 1, (3000 * data_times, features_num))  # 样本总数
#     noise = np.random.normal(0, 1, (2000 * data_times, 1))  # noise总数
#     A = generate_matrix_a()
#     z = np.dot(x, np.transpose(A))
#     y = (z[:, 2] + 0.5 * z[:, 3]) / (1 + (z[:, 0]) ** 2) + \
#         np.exp(0.5 * (z[:, 2] - z[:, 3])) * np.sin((z[:, 1] - z[:, 4] + 1.5 * z[:, 2])) + \
#         0.5 * z[:, 1] * z[:, 1] - z[:, 0] * z[:, 4] + z[:, 2]
#     y = np.reshape(y, (-1, 1))
#     y = np.concatenate((y[0:2000 * data_times] + 0.1 * noise, y[2000 * data_times:]), axis=0)
#     return torch.FloatTensor(x), torch.FloatTensor(y),A
# ######

##log21-24###
# features_num = 10
# def get_data(random_seed=123, data_times=1, features_num=10):
#     np.random.seed(random_seed)
#     x = np.random.uniform(-1, 1, (3000 * data_times, features_num))  # 样本总数
#     noise = np.random.normal(0, 1, (2000 * data_times, 1))  # noise总数
#
#     A = np.array([np.array([1,2,3,4,0,0,0,0,0,0])/np.sqrt(30), np.array([-2,1,-4,3,1,2,0,0,0,0])/np.sqrt(35),
#                   np.array([0,0,0,0,2,-1,2,1,2,1])/np.sqrt(15), np.array([0,0,0,0,0,0,-1,-1,1,1])/2])
#
#     z = np.dot(x, np.transpose(A))
#     y = (z[:, 0] * (z[:, 1]) ** 2) + (z[:, 2] * z[:, 3])
#     y = np.reshape(y, (-1, 1))
#     y = np.concatenate((y[0:2000 * data_times] + 0.5 * noise, y[2000 * data_times:]), axis=0)
#     return torch.FloatTensor(x), torch.FloatTensor(y),A
#######

# ##log25-31###
# features_num = 20
# def generate_matrix_a():
#     W = ([1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
#          [0, 0, 0, 1, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 1, 0, 0, 0, 0, 0])
#     W = np.hstack((np.array(W), np.zeros((5, features_num - 10))))  #从W的5维扩充到20维
#     V = ortho_group.rvs(20) #随机生成20x20的正交矩阵
#     A = np.dot(W, V)
#     return A
#
# def get_data(random_seed=123, data_times=1, features_num=20):
#     np.random.seed(random_seed)
#     x = np.random.uniform(-1, 1, (5000 * data_times, features_num))  # 样本总数
#     noise = np.random.normal(0, 1, (4000 * data_times, 1))  # noise总数
#
#     A = generate_matrix_a()
#     z = np.dot(x, np.transpose(A))
#     y = z[:, 4] / (5 + (1 - 0.2 * z[:, 2]) ** 2) + np.exp(0.5 * z[:, 0] + z[:, 1]) + 2 * z[:, 0] * z[:, 3] * z[:, 4] + \
#         (z[:, 3] - 0.5 * z[:, 0] + z[:, 1]) * np.cos(0.5 * z[:, 2]) + 0.2 * np.sin(z[:, 0] +z[:, 4])
#     y = np.reshape(y, (-1, 1))
#     y = np.concatenate((y[0:4000 * data_times] + 0.1 * noise, y[4000 * data_times:]), axis=0)
#     return torch.FloatTensor(x), torch.FloatTensor(y), A
# ######

# ##log32-46###
features_num = 10
def generate_matrix_a():
    A = np.zeros((4, features_num))
    A[0][0] = 1. / np.sqrt(30)
    A[0][1] = 2. / np.sqrt(30)
    A[0][2] = 3. / np.sqrt(30)
    A[0][3] = 4. / np.sqrt(30)
    A[1][0] = -2. / np.sqrt(35)
    A[1][1] = 1. / np.sqrt(35)
    A[1][2] = -4. / np.sqrt(35)
    A[1][3] = 3. / np.sqrt(35)
    A[1][4] = 1. / np.sqrt(35)
    A[1][5] = 2. / np.sqrt(35)
    A[2][4] = 2. / np.sqrt(15)
    A[2][5] = -1. / np.sqrt(15)
    A[2][6] = 2. / np.sqrt(15)
    A[2][7] = 1. / np.sqrt(15)
    A[2][8] = 2. / np.sqrt(15)
    A[2][9] = 1. / np.sqrt(15)
    A[3][6] = -1. / 2.
    A[3][7] = -1. / 2.
    A[3][8] = 1. / 2.
    A[3][9] = 1. / 2.
    # V = ortho_group.rvs(features_num) #随机生成20x20的正交矩阵
    # A = np.dot(W, V)
    return A

def get_data(random_seed=123, data_times=1, features_num=10):
    np.random.seed(random_seed)
    x = np.random.normal(0, 1, (2000 * data_times, features_num))  # 样本总数
    # x = np.random.uniform(-1, 1, (3000 * data_times, features_num))
    noise = np.random.normal(0, 1, (1000 * data_times, 1))  # noise总数
    A = generate_matrix_a()
    z = np.dot(x, np.transpose(A))
    # y = 0.5 * z[:, 0] * z[:, 1] + np.sin(z[:, 0] - z[:, 2]) + np.cos(z[:, 1] + z[:, 2])
    y = z[:, 0] * z[:, 1]**2 + z[:, 2] * z[:, 3]
    y = np.reshape(y, (-1, 1))
    y = np.concatenate((y[0:1000 * data_times] + 0.5 * noise, y[1000 * data_times:]), axis=0)
    return torch.FloatTensor(x), torch.FloatTensor(y), A
# ######

if __name__ == '__main__':
    x, y, beta = get_data(random_seed=123, data_times=1)
    #save_data_to_path(x, y, "./data/")
    #save_matrix2csv(beta, "./data/")
    pass
