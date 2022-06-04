import numpy as np
from sklearn.cluster import KMeans
from tools.distance import get_distance_list


class Cluster:

    def __init__(self, data, classname):
        self.data = data
        self.distance_graph = get_distance_list(self.data)
        self.classname = classname
        self.distance_graph = np.array(self.distance_graph)
        self.num_clusters = int(pow(len(self.data) / 2, 0.5))  # 初始化类簇个数
        self.centroids_list, self.local_best_path = self.ClusterTrain()  # 获得质心连接顺序，每个类内的连接顺序

    def ClusterTrain(self):
        # 训练模型
        cluster_node = KMeans(n_clusters=self.num_clusters)  # 生成聚类实体
        cluster_node.fit(self.data)  # 对数据self.data进行聚类分析
        label = cluster_node.fit_predict(self.data)  # 获取label结果（标签）
        centroids = cluster_node.cluster_centers_  # 获取质心坐标

        # 按标签将点分别存放
        city_cluster = [[] for i in range(self.num_clusters)]
        for i in range(len(self.data)):
            city_cluster[label[i]].append(self.data[i])

        # 确认不同类簇之间的连接顺序
        centroids_dis = get_distance_list(centroids)
        centroids_tsp = self.classname(centroids, centroids_dis, 100)
        centroids_list = centroids_tsp.train()

        # 对每个区域的点进行路径规划
        best_path1 = []
        for i in range(self.num_clusters):
            city_cluster1 = np.array(city_cluster[i])
            city_cluster1_dis = get_distance_list(city_cluster1)

            ga1 = self.classname(city_cluster1, city_cluster1_dis)
            best_path = ga1.train()

            # 局部路径转化为全局路径
            temp = city_cluster1[best_path[0:-1]]
            temp2 = list(map(int, temp[:, 2]))
            best_path1.append(temp2)

        return centroids_list, best_path1

    def train(self):
        temp_node1 = self.local_best_path[self.centroids_list[0]]

        # 查找类簇间距离最近的点
        for i in range(self.num_clusters - 1):
            min_list = []
            temp_node2 = self.local_best_path[self.centroids_list[i + 1]]

            for j in range(len(temp_node2)):
                test_dis1 = min(self.distance_graph[temp_node2[j], temp_node1])
                min_list.append(test_dis1)
            min_list.sort()
            if min_list[0] == min_list[1]:
                min1 = min_list[0]
            else:
                min1 = min_list[1]

            for j in range(1, len(temp_node1) - 1):
                for k in range(1, len(temp_node2) - 1):
                    if self.distance_graph[temp_node1[j]][temp_node2[k]] == min1:
                        if self.distance_graph[temp_node1[j]][temp_node2[k]] == \
                                self.distance_graph[temp_node1[j - 1]][temp_node2[k - 1]]:
                            min_id11, min_id12, min_id21, min_id22 = j, j - 1, k, k - 1
                            connect_label = 1
                            break
                        if self.distance_graph[temp_node1[j]][temp_node2[k]] == \
                                self.distance_graph[temp_node1[j - 1]][temp_node2[k + 1]]:
                            min_id11, min_id12, min_id21, min_id22 = j, j - 1, k, k + 1
                            connect_label = 2
                            break
                        if self.distance_graph[temp_node1[j]][temp_node2[k]] == \
                                self.distance_graph[temp_node1[j + 1]][temp_node2[k + 1]]:
                            min_id11, min_id12, min_id21, min_id22 = j, j + 1, k, k + 1
                            connect_label = 3
                            break
                        if self.distance_graph[temp_node1[j]][temp_node2[k]] == \
                                self.distance_graph[temp_node1[j + 1]][temp_node2[k - 1]]:
                            min_id11, min_id12, min_id21, min_id22 = j, j + 1, k, k - 1
                            connect_label = 4
                            break

            fragment11 = temp_node2[0:min_id21]
            fragment11 = fragment11[::-1]  # 翻转数组
            fragment12 = temp_node2[min_id22 + 1:]
            fragment12 = fragment12[::-1]

            fragment31 = temp_node2[0:min_id21 + 1]
            fragment31 = fragment31[::-1]  # 翻转数组
            fragment32 = temp_node2[min_id22:]
            fragment32 = fragment32[::-1]

            if connect_label == 1:
                connect_result = temp_node1[0:min_id11] + fragment11 + \
                                 fragment12 + temp_node1[min_id12 + 1:]

            if connect_label == 2:
                connect_result = temp_node1[0:min_id11] + temp_node2[min_id22:] + \
                                 temp_node2[:min_id21 + 1] + temp_node1[min_id12 + 1:]

            if connect_label == 3:
                connect_result = temp_node1[0:min_id11 + 1] + fragment31 + \
                                 fragment32 + temp_node1[min_id12:]

            if connect_label == 4:
                connect_result = temp_node1[0:min_id11 + 1] + temp_node2[min_id22 + 1:] + \
                                 temp_node2[:min_id21] + temp_node1[min_id12:]

            temp_node1 = connect_result
            connect_result.append(connect_result[0])

        return self.data[connect_result]  # 返回排序后的坐标
