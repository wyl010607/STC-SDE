import tqdm
import os
import numpy as np
from fastdtw import fastdtw
def get_Adj_matrix(
        data_,
        dwt_path,
        sigma1,
        thres1,
):
    num_node = data_.shape[1]
    mean_value = np.mean(data_, axis=(0)).reshape(1, -1)
    std_value = np.std(data_, axis=(0)).reshape(1, -1)
    mean_value = mean_value.reshape(-1)[0]
    std_value = std_value.reshape(-1)[0]

    if not os.path.exists(dwt_path):
        data_mean = np.mean(
            [data_[:, :][24 * 12 * i: 24 * 12 * (i + 1)] for i in range(data_.shape[0] // (24 * 12))], axis=0)
        data_mean = data_mean.squeeze().T
        dtw_distance = np.zeros((num_node, num_node))
        for i in tqdm(range(num_node)):
            for j in range(i, num_node):
                dtw_distance[i][j] = fastdtw(data_mean[i], data_mean[j], radius=6)[0]
        for i in range(num_node):
            for j in range(i):
                dtw_distance[i][j] = dtw_distance[j][i]
        np.save(dwt_path, dtw_distance)

    dist_matrix = np.load(dwt_path)
    mean = np.mean(dist_matrix)
    std = np.std(dist_matrix)
    dist_matrix = (dist_matrix - mean) / std
    sigma = sigma1
    thres = thres1
    dist_matrix = np.exp(-dist_matrix ** 2 / sigma ** 2)
    dtw_matrix = np.zeros_like(dist_matrix)
    dtw_matrix[dist_matrix > thres] = 1