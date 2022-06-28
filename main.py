import path.base_algo
from tools import getdate as gd
from path import path_cluster, advanced_algo, base_algo
from tools.distance import *
from tools.visualization_type import *
import time
import test3

if __name__ == '__main__':
    file_path = "data/E-204-1单甲醚塔冷却器 管板Ⅰ.csv"
    start_time = time.time()
    index, r0 = gd.get_data(file_path)
    distance_graph = get_distance_list(index)

    # 不使用聚类
    # class1 = path.base_algo.PSO(index)
    # class1 = path.advanced_algo.Ga01(distance_graph)
    # class1 = path.advanced_algo.IGSA(distance_graph)
    # class1 = path.advanced_algo.TS(distance_graph)
    # result_path, result_length = class1.run()
    # end_time = time.time()
    # sum_time = end_time - start_time
    # result_path = index[result_path]
    # result_lengths = class1.best_lengths
    # draw_path(index, result_path, result_length, sum_time)
    # draw_dis_list(result_lengths)

    # 使用聚类方法
    # classname = path.base_algo.TwoOpt  # 聚类内使用的优化算法
    classname = path.advanced_algo.TS
    class1 = path_cluster.Cluster(index, classname)
    result_path = class1.run()
    result_length = get_distance0(result_path)
    end_time = time.time()
    sum_time = end_time - start_time
    draw_path(index, result_path, result_length, sum_time)

    # 依次连接，查看最短路径
    # class1 = base_algo.RowCol(index)
    # result_path = class1.run()
    # draw_path(index, result_path, sum_time)
    # print(get_distance0(result_path))
