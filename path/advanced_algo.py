import numpy as np
import config
from path.path_common import *


class TwoOpt:
    def __init__(self, distance_graph, max_iter=config.MAXCOUNT):
        self.distance_graph = distance_graph
        self.max_iter = max_iter

    def calPathDist(self, indexList):
        sum_distance = 0.0
        for i in range(1, len(indexList)):
            sum_distance += self.distance_graph[indexList[i]][indexList[i - 1]]
        return sum_distance

    # 比较路径段，path1长度比path2短则返回true
    def path_compare(self, path1, path2):
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
            if self.path_compare(path, rePath):
                count += 1
                continue
            else:
                count = 0
                best_path[start:end + 1] = rePath
        return best_path

    def train(self):
        # 随便选择一条可行路径
        best_path = np.arange(0, len(self.distance_graph))
        best_path = np.append(best_path, 0)
        best_path = self.update_best_path(best_path)

        return best_path


class Ga01:

    def __init__(self, distance_graph, max_iter=config.max_iter):
        self.distance_graph = np.array(distance_graph)
        self.chromosome_length = len(self.distance_graph[0])
        self.population_size = int(1.5 * self.chromosome_length)
        self.max_iter = max_iter
        self.population = []
        self.fitness_value = []
        self.best_path = []
        self.best_dis_list = []
        self.best_total_distance = 0
        self.global_best_path = []

    # 使用贪婪算法产生初始种群

    def create_origin_population(self):
        self.population = [
            [-1 for col in range(self.chromosome_length)] for col in range(self.population_size)]
        for i in range(self.population_size):
            temp = create_origin_greedy(self.distance_graph)
            self.population[i] = temp

    # 计算适应度
    def eva_fitness(self):
        self.fitness_value = []
        for i in range(self.population_size):
            temp = eva_fitness0(self.distance_graph, self.population[i])
            self.fitness_value.append(1. / temp)

        # 选择适应度最大的个体
        temp_max = self.fitness_value[0]
        temp_index = 0
        for j in range(self.population_size):
            if self.fitness_value[j] > temp_max:
                temp_max = self.fitness_value[j]
                temp_index = j
        self.best_path = self.population[temp_index]
        self.best_total_distance = eva_fitness0(self.distance_graph, self.best_path)

    # 交叉
    # 部分匹配交叉PMX
    # 交叉
    def cross(self):
        random.shuffle(self.population)
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
                    self.population[i] = gene_temp

    # 选择
    # 轮盘赌方式
    def select(self):
        # 随机选取n个个体复制到下一代种群内
        next_population = []
        for i in range(self.population_size):
            temp = roulette_wheel_selection(self.fitness_value)
            next_population.append(self.population[temp])
        next_population[0] = self.global_best_path
        self.population = next_population

    def next_gen(self):
        self.cross()
        self.mutate()
        self.select()

    def run(self):
        self.create_origin_population()
        self.eva_fitness()
        self.global_best_path = self.best_path
        for i in range(self.max_iter):
            self.next_gen()
            self.eva_fitness()
            self.best_dis_list.append(self.best_total_distance)
        self.best_path.append(self.best_path[0])
        return self.best_path, self.best_dis_list


class IGSA:
    def __init__(self, distance_graph, max_iter=config.max_iter):
        self.global_best_path = None
        self.distance_graph = distance_graph
        self.chromosome_length = len(self.distance_graph[0])
        self.population_size = 50
        self.max_iter = max_iter
        self.fitness_value = []
        self.best_path = []
        self.population = [random.sample(range(self.chromosome_length), self.chromosome_length) for col in
                           range(self.population_size)]
        self.template_initial = config.template_initial
        self.decrease = config.decrease
        self.template_end = config.template_end
        self.best_total_distance = []
        self.next_population = []
        self.best_path_list = []

    # 计算适应度
    def eva_fitness(self):
        self.fitness_value = []
        for i in range(self.population_size):
            temp = eva_fitness0(self.distance_graph, self.population[i])
            self.fitness_value.append(1. / temp)

        # 选择适应度最大的个体
        temp_max = self.fitness_value[0]
        temp_index = 0
        for j in range(self.population_size):
            if self.fitness_value[j] > temp_max:
                temp_max = self.fitness_value[j]
                temp_index = j
        self.best_path = self.population[temp_index]
        self.best_total_distance = eva_fitness0(self.distance_graph, self.best_path)

    # 交叉
    # 部分匹配交叉PMX
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

    # 遗传变异
    def mutate1(self):
        for i in range(self.population_size):
            if random.random() < config.prob_mutate:
                gene_temp = mutate_reverse(self.population[i][:])
                temp = 0
                for j in range(self.chromosome_length - 1):
                    temp += self.distance_graph[gene_temp[j + 1]][gene_temp[j]]
                t1 = (1. / temp)
                t2 = self.fitness_value[i]
                if t1 > t2:
                    self.population[i] = gene_temp

    # 模拟退火变异
    def mutateSa(self):
        gene_temp = mutate_reverse(self.population)
        self.next_population = gene_temp

    # 选择
    # 轮盘赌方式
    def select(self):
        # 随机选取n个个体复制到下一代种群内
        next_population = []
        for i in range(self.population_size):
            temp = roulette_wheel_selection(self.fitness_value)
            next_population.append(self.population[temp])
        next_population[0] = self.global_best_path
        self.population = next_population

    def metropolis(self):
        result1 = eva_fitness0(self.distance_graph, self.population)
        result2 = eva_fitness0(self.distance_graph, self.next_population)
        result = Metropolis0(result2, result1, self.template_now)
        if result:
            self.population = self.next_population

    def run(self):
        self.eva_fitness()
        self.global_best_path = self.best_path
        for i in range(self.max_iter):
            self.eva_fitness()
            self.select()
            self.cross()
            self.mutate1()
            self.best_path_list.append(self.best_total_distance)
        self.population = self.best_path
        self.template_now = self.template_initial
        while self.template_now > self.template_end:
            for i in range(self.max_iter):
                self.mutateSa()
                self.metropolis()
                self.best_path_list.append(eva_fitness0(self.distance_graph, self.population))
            self.template_now *= self.decrease
        self.best_path.append(self.best_path[0])
        return self.population, self.best_path_list
