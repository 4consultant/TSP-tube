import config
import numpy as np
import random

from path.path_common import *


class RowCol:
    def __init__(self, index):
        self.index = index[:, 0:2]
        self.total_path = self.index

    def train(self):
        self.index = self.index[self.index[:, 0].argsort()]
        index1 = 0
        label = 0
        for i in range(1, len(self.index)):
            if self.index[i, 0] != self.index[i - 1, 0] or i == len(self.index) - 1:
                temp = self.index[index1:i]
                temp = temp[temp[:, 1].argsort()]
                if label % 2:
                    temp = temp[::-1]
                self.total_path[index1:i] = temp
                index1 = i
                label += 1
        self.total_path = self.total_path[0:-1]
        return self.total_path


# 基本遗传算法
class GaBase:

    def __init__(self, distance_graph, max_iter=config.max_iter):
        self.distance_graph = np.array(distance_graph)
        self.chromosome_length = len(self.distance_graph[0])
        self.population_size = 50
        self.max_iter = max_iter
        self.population = []
        self.fitness_value = []
        self.best_path = []
        self.best_dislist = []
        self.best_total_distance = 0
        self.create_origin_population()

    # 产生初始种群
    def create_origin_population(self):
        self.population = [
            random.sample(range(self.chromosome_length), self.chromosome_length) for col in
            range(self.population_size)]

    # 计算适应度
    def eva_fitness(self):
        self.fitness_value = []
        for i in range(self.population_size):
            temp = eva_fitness0(self.distance_graph, self.population[i])
            self.fitness_value.append(1. / temp)
        return self.fitness_value

    # 交叉
    def cross(self):
        # random.shuffle(self.population)
        for i in range(self.population_size - 1, 2):
            genes1 = self.population[i][:]
            genes2 = self.population[i + 1][:]

            if random.random() < config.prob_cross:
                genes3, genes4 = cross1(genes1, genes2)
                self.population[i] = genes3
                self.population[i + 1] = genes4

    # 变异
    def mutate(self):
        for i in range(self.population_size):
            if random.random() < config.prob_mutate:
                gene_temp = mutate_reverse(self.population[i][:])
                temp = 0
                for j in range(self.chromosome_length - 1):
                    temp += self.distance_graph[gene_temp[j + 1]][gene_temp[j]]
                t1 = (1. / temp)
                t2 = self.fitness_value[i]
                if t1 > t2:
                    self.population[i][:] = gene_temp

    # 选择
    # 轮盘赌方式
    def select(self):
        new_fitness = []
        total_fitness = sum(self.fitness_value)
        # 按比例适应度分配
        for i in range(self.population_size):
            new_fitness.append(self.fitness_value[i] / total_fitness)

        # 选择适应度最大的个体
        for i in range(self.population_size):
            if new_fitness[i] == max(new_fitness):
                self.best_path = self.population[i][:]

        # 随机选取n个个体复制到下一代种群内
        next_population = self.population
        for i in range(self.population_size):
            temp = select_rws(new_fitness)
            next_population[i] = self.population[temp]

        self.population = next_population

    def next_gen(self):
        self.eva_fitness()
        self.cross()
        self.mutate()
        self.select()

    def train(self):
        for i in range(self.max_iter):
            self.next_gen()
            self.best_dislist.append(self.best_total_distance)
        self.best_path.append(self.best_path[0])
        return self.best_path
