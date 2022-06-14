import numpy as np


# 获取路径的总距离
def get_distance(distance_graph, list1):
    #list1 = list1[0:-1]
    total_distance = 0
    for i in range(len(list1) - 1):
        total_distance += distance_graph[list1[i+1]][list1[i]]

    return total_distance


# 获取两点之间的距离
def two_points_dis(x, y):
    return (np.sum(np.power(x[0] - y[0], 2) + np.power(x[1] - y[1], 2))) ** 0.5


# 获取点的距离表
def get_distance_list(data):
    distance_graph = [[0.0 for col in range(
        len(data))] for raw in range(len(data))]

    for i in range(len(data)):
        for j in range(len(data)):
            temp_distance = pow(
                (data[i, 0] - data[j, 0]), 2) + pow((data[i, 1] - data[j, 1]), 2)
            temp_distance = pow(temp_distance, 0.5)
            distance_graph[i][j] = float(int(temp_distance + 0.5))
            if i == j:
                distance_graph[i][j] = 1000000
    return distance_graph
