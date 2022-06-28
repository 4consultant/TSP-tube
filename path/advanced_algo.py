import numpy as np
import config
from path.path_common import *


class Ga01:

    def __init__(self, distance_graph, max_iter=config.max_iter):
        self.distance_graph = np.array(distance_graph)
        self.chromosome_length = len(self.distance_graph[0])
        self.population_size = int(1.5 * self.chromosome_length)
        self.max_iter = max_iter
        self.population = []
        self.fitness_value = []
        self.best_path = []
        self.best_lengths = []
        self.best_length = 0
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
        self.best_length = eva_fitness0(self.distance_graph, self.best_path)

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
        self.select()
        self.cross()
        self.mutate()

    def run(self):
        self.create_origin_population()
        self.eva_fitness()
        self.global_best_path = self.best_path
        for i in range(self.max_iter):
            self.next_gen()
            self.eva_fitness()
            if eva_fitness0(self.distance_graph, self.global_best_path) > self.best_length:
                self.global_best_path = self.best_path
            self.best_lengths.append(self.best_length)
        self.best_path.append(self.best_path[0])
        return self.best_path, self.best_length


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
        self.best_length = 0
        self.next_population = []
        self.best_lengths = []
        self.template_now = 0

    # 计算适应度
    def eva_fitness(self):
        self.fitness_value = []
        for i in range(self.population_size):
            temp = eva_fitness0(self.distance_graph, self.population[i])
            self.fitness_value.append(temp)
        temp_max = max(self.fitness_value)
        for j in range(self.population_size):
            temp = 1 - self.fitness_value[j] / temp_max
            self.fitness_value.append(temp)

        # 选择适应度最大的个体
        temp_max = self.fitness_value[0]
        temp_index = 0
        for j in range(self.population_size):
            if self.fitness_value[j] > temp_max:
                temp_max = self.fitness_value[j]
                temp_index = j
        self.best_path = self.population[temp_index]
        self.best_length = eva_fitness0(self.distance_graph, self.best_path)

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
            self.best_lengths.append(self.best_length)
        self.population = self.best_path
        self.template_now = self.template_initial
        while self.template_now > self.template_end:
            for i in range(self.max_iter):
                self.mutateSa()
                self.metropolis()
                self.best_lengths.append(eva_fitness0(self.distance_graph, self.population))
            self.template_now *= self.decrease
        self.population.append(self.population[0])
        self.best_length = self.best_lengths[-1]
        return self.population, self.best_length


# 改进变异算子（未完成）
class Ga02:

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
        self.k = config.GAK

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

    # 变异HMO
    def mutateHMO(self):
        flag = True
        while flag:
            for i in range(self.population_size):
                if random.random() < config.prob_mutate:
                    temp_listX = self.population[i]
                    temp_listX_index = i
                    break
            temp_index = random.randint(0, self.chromosome_length)

            path_index = temp_listX[temp_index]
            sort_path = sorted(self.distance_graph[path_index])
            k_path = sort_path[:self.k]  # 距离点path_index最近的k个点
            temp_kdis = []  # 距离点path_index最近的k个点的距离

            for i in range(self.k):
                temp_kdis.append(1. / self.distance_graph[path_index][k_path[i]])
            temp_result_index = roulette_wheel_selection(temp_kdis)
            temp_result = k_path[temp_result_index]

            if temp_result != temp_listX[temp_index - 1] and temp_result != temp_listX[temp_index + 1]:
                flag = False
        lambda1 = self.distance_graph[temp_listX[temp_result_index - 1]][temp_listX[temp_index]] \
                  + self.distance_graph[temp_listX[temp_result_index]][temp_listX[temp_index]] \
                  + self.distance_graph[temp_listX[temp_index - 1]][temp_listX[temp_index + 1]] \
                  - self.distance_graph[temp_listX[temp_index - 1]][temp_listX[temp_index]] \
                  - self.distance_graph[temp_listX[temp_index]][temp_listX[temp_index + 1]] \
                  - self.distance_graph[temp_listX[temp_result_index - 1]][temp_listX[temp_result_index]]
        lambda2 = self.distance_graph[temp_listX[temp_result_index]][temp_listX[temp_index]] \
                  + self.distance_graph[temp_listX[temp_result_index + 1]][temp_listX[temp_index]] \
                  + self.distance_graph[temp_listX[temp_index - 1]][temp_listX[temp_index + 1]] \
                  - self.distance_graph[temp_listX[temp_index - 1]][temp_listX[temp_index]] \
                  - self.distance_graph[temp_listX[temp_index]][temp_listX[temp_index + 1]] \
                  - self.distance_graph[temp_listX[temp_result_index]][temp_listX[temp_result_index + 1]]
        if lambda1 <= lambda2:
            pass

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
        self.select()
        self.cross()
        self.mutate()

    def run(self):
        self.create_origin_population()
        self.eva_fitness()
        self.global_best_path = self.best_path
        for i in range(self.max_iter):
            self.next_gen()
            self.eva_fitness()
            if eva_fitness0(self.distance_graph, self.global_best_path) > self.best_total_distance:
                self.global_best_path = self.best_path
            self.best_dis_list.append(self.best_total_distance)
        self.best_path.append(self.best_path[0])
        return self.best_path, self.best_dis_list


# 贪婪禁忌搜索
class TS(object):
    def __init__(self, distance_graph, maxiter=config.max_iter):
        self.taboo_size = 5
        self.iteration = maxiter
        self.distance_graph = distance_graph
        self.num_city = len(self.distance_graph)
        self.taboo = []

        self.path = create_origin_greedy(self.distance_graph)
        self.best_path = self.path
        self.cur_path = self.path
        self.best_length = self.compute_pathlen(self.path, self.distance_graph)

        # 显示初始化后的路径
        init_pathlen = 1. / self.compute_pathlen(self.path, self.distance_graph)
        # 存储结果，画出收敛图
        self.iter_x = [0]
        self.best_lengths = [1. / init_pathlen]

    # 计算路径长度
    def compute_pathlen(self, path, dis_mat):
        a = path[0]
        b = path[-1]
        result = dis_mat[a][b]
        for i in range(len(path) - 1):
            a = path[i]
            b = path[i + 1]
            result += dis_mat[a][b]
        return result

    # 计算一个群体的长度
    def compute_paths(self, paths):
        result = []
        for one in paths:
            length = self.compute_pathlen(one, self.distance_graph)
            result.append(length)
        return result

    # 产生随机解
    def ts_search(self, x):
        moves = []
        new_paths = []
        while len(new_paths) < 400:
            i = np.random.randint(len(x))
            j = np.random.randint(len(x))
            tmp = x.copy()
            tmp[i:j] = tmp[i:j][::-1]
            new_paths.append(tmp)
            moves.append([i, j])
        return new_paths, moves

    # 禁忌搜索
    def ts(self):
        for cnt in range(self.iteration):
            new_paths, moves = self.ts_search(self.cur_path)
            new_lengths = self.compute_paths(new_paths)
            sort_index = np.argsort(new_lengths)
            min_l = new_lengths[sort_index[0]]
            min_path = new_paths[sort_index[0]]
            min_move = moves[sort_index[0]]

            # 更新当前的最优路径
            if min_l < self.best_length:
                self.best_length = min_l
                self.best_path = min_path
                self.cur_path = min_path
                # 更新禁忌表
                if min_move in self.taboo:
                    self.taboo.remove(min_move)

                self.taboo.append(min_move)
            else:
                # 找到不在禁忌表中的操作
                while min_move in self.taboo:
                    sort_index = sort_index[1:]
                    min_path = new_paths[sort_index[0]]
                    min_move = moves[sort_index[0]]
                self.cur_path = min_path
                self.taboo.append(min_move)
            # 禁忌表超长了
            if len(self.taboo) > self.taboo_size:
                self.taboo = self.taboo[1:]
            self.iter_x.append(cnt)
            self.best_lengths.append(self.best_length)

    def run(self):
        self.ts()
        self.best_path.append(self.best_path[0])
        return self.best_path, self.best_length
