from tools import getdate as gd
from path import path_cluster, advanced_algo
from tools.distance import *
from tools.visualization_type import *
import time


if __name__ == '__main__':
    file_path = "data/E-204-1单甲醚塔冷却器 管板Ⅰ.csv"
    classname = advanced_algo.TwoOpt  # 聚类内使用的优化算法

    start_time = time.time()
    index, r0 = gd.get_data(file_path)
    distance_graph = get_distance_list(index)

    # class1 = Path.RowCol(index)  # 不使用聚类
    # class1 = Path.TwoOpt(index, distance_graph)
    class1 = path_cluster.Cluster(index, classname)  # 使用聚类

    result = class1.train()
    end_time = time.time()
    sum_time = end_time - start_time
    draw(index, result, sum_time)
    print("最佳路径的总距离：", get_distance(result))
