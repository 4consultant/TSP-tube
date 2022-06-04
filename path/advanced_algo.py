import numpy as np
import config
from path.path_common import *


def generate_random_path(best_path):
    a = np.random.randint(len(best_path))
    while True:
        b = np.random.randint(len(best_path))
        if np.abs(a - b) > 1:
            break
    if a > b:
        return b, a, best_path[b:a + 1]
    else:
        return a, b, best_path[a:b + 1]


class TwoOpt:
    def __init__(self, coordinates, distance_graph, max_iter=config.MAXCOUNT):
        self.cities = coordinates
        self.distance_graph = distance_graph
        self.max_iter = max_iter

    def calDist(self, xindex, yindex):
        return (np.sum(np.power(self.cities[xindex] - self.cities[yindex], 2))) ** 0.5

    def calPathDist(self, indexList):
        sum_distance = 0.0
        for i in range(1, len(indexList)):
            sum_distance += self.distance_graph[indexList[i]][indexList[i - 1]]
        return sum_distance

    # path1长度比path2短则返回true
    def pathCompare(self, path1, path2):
        if self.calPathDist(path1) <= self.calPathDist(path2):
            return True
        return False

    def update_best_path(self, best_path):
        count = 0
        while count < self.max_iter:
            # print(self.calPathDist(best_path))
            # print(best_path.tolist())
            start, end, path = generate_random_path(best_path)
            rePath = reverse_path(path)
            if self.pathCompare(path, rePath):
                count += 1
                continue
            else:
                count = 0
                best_path[start:end + 1] = rePath
        return best_path

    def train(self):
        # 随便选择一条可行路径
        best_path = np.arange(0, len(self.cities))
        best_path = np.append(best_path, 0)
        best_path = self.update_best_path(best_path)

        return best_path
