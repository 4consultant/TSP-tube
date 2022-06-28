import math
import random
import numpy as np
import config
import copy
from tools.distance import *


# 翻转路径段
def reverse_path(path):
    re_path = path.copy()
    re_path[1:-1] = re_path[-2:0:-1]
    return re_path


# 从路径中随机选取一段路径，返回起始位置、结尾位置、选中的路径段
def generate_random_path(path):
    a = np.random.randint(len(path))
    while True:
        b = np.random.randint(len(path))
        if np.abs(a - b) > 1:
            break
    if a > b:
        return b, a, path[b:a + 1]
    else:
        return a, b, path[a:b + 1]


# 使用贪心算法生成初始种群
def create_origin_greedy(distance_graph):
    t1 = random.randint(0, len(distance_graph) - 1)
    flag = [0 for i in range(len(distance_graph))]
    population = [-1 for i in range(len(distance_graph))]
    population[0] = t1
    flag[t1] = 1
    for i in range(0, len(distance_graph) - 1):
        min1 = 100000
        for j in range(len(distance_graph)):
            if flag[j]:
                continue
            if distance_graph[population[i]][j] < min1:
                min1 = distance_graph[population[i]][j]
                temp1 = j
        flag[temp1] = 1
        population[i + 1] = temp1

    return population


# 计算适应度
def eva_fitness0(distance_graph, path_list):
    temp = 0
    for i in range(len(path_list) - 1):
        temp += distance_graph[path_list[i + 1]][path_list[i]]
    temp += distance_graph[path_list[-1]][path_list[0]]
    return temp


# 交叉
def cross1(path_list1, path_list2):
    genes1 = path_list1
    genes2 = path_list2

    index1 = random.randint(0, len(path_list1) - 2)
    index2 = random.randint(index1 + 1, len(path_list1) - 1)
    pos1_recorder = {value: idx for idx, value in enumerate(genes1)}
    pos2_recorder = {value: idx for idx, value in enumerate(genes2)}
    for i in range(index1, index2 + 1):
        value1, value2 = genes1[i], genes2[i]
        pos1, pos2 = pos1_recorder[value2], pos2_recorder[value1]
        genes1[i], genes1[pos1] = genes1[pos1], genes1[i]
        genes2[i], genes2[pos2] = genes2[pos2], genes2[i]
        pos1_recorder[value1], pos1_recorder[value2] = pos1, i
        pos2_recorder[value1], pos2_recorder[value2] = i, pos2

    return genes1, genes2


# 翻转变异
def mutate_reverse(path_list):
    old_gen = path_list

    index1 = random.randint(0, len(path_list) - 2)
    index2 = random.randint(index1, len(path_list) - 1)

    gene_mutate = old_gen[index1:index2]
    gene_mutate.reverse()
    gene_temp = old_gen[:index1] + gene_mutate + old_gen[index2:]

    return gene_temp


# 选择
# 轮盘赌方式(正则化适应度列表)
def roulette_wheel_selection(fitness_list):
    wheel = sum(fitness_list)
    pick = np.random.uniform(0, wheel)
    current = 0
    for i in range(len(fitness_list)):
        current += fitness_list[i]
        if current > pick:
            result = i
            break
    return result


def sum_list(fitness_list):
    total = 0
    for i in range(len(fitness_list)):
        total += fitness_list[i]
    return total


def Metropolis0(distance_new, distance_old, T):
    # result=1,代表用新路径代替老路径
    if distance_new < distance_old:
        result = 1
    if distance_new >= distance_old:
        temp = math.exp((distance_old - distance_new) / T)  # 距离增加量的越大，概率越小
        temp1 = random.random()
        if temp >= temp1:
            result = 1
        if temp < temp1:
            result = 0
    return result


'''
def HMO(distance_graph, path_index, k):
    sort_path = sorted(distance_graph[path_index])
    k_path = sort_path[:k]  # 距离点path_index最近的k个点
    temp_kdis = 0  # 距离点path_index最近的k个点的距离之和
    prod_c = []  # 距离点path_index最近的k个点的距离与距离之和的比值
    prod_csum = []  # 距离点path_index最近的k个点的距离的依次累加值
    for i in range(k):
        temp_kdis += (1. / distance_graph[path_index][k_path[i]])
    for j in range(k):
        prod_c.append((1. / distance_graph[path_index][k_path[j]]) / temp_kdis)   
    for n in range(k):
        temp = 0
        for m in range(n):
            temp += prod_c[m]
            prod_csum.append(temp)
    r = random.random()
    if r <= prod_csum[0]:
        temp_result = k_path[0]
    for h in range(k):
        if prod_csum[h - 1] < r <= prod_csum[h]:
            temp_result = k_path[h]
    
    return temp_result
'''
