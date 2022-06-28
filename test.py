import numpy as np
from path.path_common import *
from tools.visualization_type import *
from tools.getdate import *
from path.advanced_algo import *
from path.base_algo import *



file_path="data/E-204-1单甲醚塔冷却器 管板Ⅰ.csv"
# index, r0 = get_data(file_path)
index = config.test14
distance_graph = get_distance_list(index)

'''
# temp = GaBase(distance_graph)
# temp = Ga01(distance_graph)
temp = IGSA(distance_graph)
temp1, temp_list = temp.run()
draw_path(index, index[temp1], 0, 1)
draw_dis_list(temp_list)
'''

# temp1 = GaBase(distance_graph)
temp1 = GaBase(distance_graph)
temp2 = Ga01(distance_graph)
temp1, temp_list1 = temp1.run()
temp2, temp_list2 = temp2.run()
draw_path(index, index[temp1], 0, 1)
draw_path(index, index[temp2], 0, 1)
# print(len(temp_list1), len(temp_list2))
lists = [temp_list1, temp_list2]
draw_dis_lists(lists)


