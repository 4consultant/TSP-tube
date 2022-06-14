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
def select_rws(fitness_list):
    total_prob = random.random()
    for i in range(len(fitness_list)):
        total_prob -= fitness_list[i]
        if total_prob < 0:
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

