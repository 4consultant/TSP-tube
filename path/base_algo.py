import config
import numpy as np
import random

from path.path_common import *


class RowCol:
    def __init__(self, index):
        self.index = index[:, 0:2]
        self.total_path = self.index

    def run(self):
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


# 2-opt算法
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

    def run(self):
        # 随便选择一条可行路径
        best_path = np.arange(0, len(self.distance_graph))
        best_path = np.append(best_path, 0)
        best_path = self.update_best_path(best_path)
        best_length = self.calPathDist(best_path)

        return best_path



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
        self.best_total_distance = 0
        self.best_dis_list = []

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
                self.population[i] = gene_temp

    # 选择
    # 轮盘赌方式
    def select(self):

        # 随机选取n个个体复制到下一代种群内
        next_population = []
        for i in range(self.population_size):
            temp = roulette_wheel_selection(self.fitness_value)
            next_population.append(self.population[temp])
        next_population[0] = self.best_path
        self.population = next_population

    def next_gen(self):
        self.select()
        self.cross()
        self.mutate()

    def run(self):
        self.create_origin_population()
        self.eva_fitness()
        for i in range(self.max_iter):
            self.next_gen()
            self.eva_fitness()
            self.best_dis_list.append(self.best_total_distance)
        self.best_path.append(self.best_path[0])
        return self.best_path, self.best_dis_list


# 基本模拟退火算法
class SaBase:

    def __init__(self, distance_graph, max_iter=config.max_iter):
        self.distance_graph = distance_graph
        self.template_initial = config.template_initial
        self.template_now = self.template_initial
        self.template_end = config.template_end
        self.decrease = config.decrease
        self.max_iter = max_iter
        self.population = random.sample(range(len(self.distance_graph)), len(self.distance_graph))
        self.next_population = self.population

    def mutate(self):
        temp = self.population.copy()
        result = mutate_reverse(temp)
        self.next_population = result

    def select(self):
        result1 = eva_fitness0(self.distance_graph, self.population)
        result2 = eva_fitness0(self.distance_graph, self.next_population)
        result = Metropolis0(result2, result1, self.template_now)
        if result:
            self.population = self.next_population

    def run(self):
        while self.template_now > self.template_end:
            for i in range(self.max_iter):
                self.mutate()
                self.select()
            self.template_now *= self.decrease
        return self.population


# 基本粒子群算法
class PSO(object):
    def __init__(self, data):
        self.iter_max = 500  # 迭代数目
        self.num = 200  # 粒子数目
        self.location = data  # 城市的位置坐标
        self.num_city = len(self.location)  # 城市数
        # 计算距离矩阵
        self.dis_mat = self.compute_dis_mat(self.num_city, self.location)  # 计算城市之间的距离矩阵
        # 初始化所有粒子
        # self.particals = self.random_init(self.num, num_city)
        self.particals = self.greedy_init(self.dis_mat, num_total=self.num, num_city=self.num_city)
        self.lenths = self.compute_paths(self.particals)
        # 得到初始化群体的最优解
        init_l = min(self.lenths)
        init_index = self.lenths.index(init_l)
        init_path = self.particals[init_index]
        # 画出初始的路径图
        init_show = self.location[init_path]
        # 记录每个个体的当前最优解
        self.local_best = self.particals
        self.local_best_len = self.lenths
        # 记录当前的全局最优解,长度是iteration
        self.global_best = init_path
        self.global_best_len = init_l
        # 输出解
        self.best_l = self.global_best_len
        self.best_path = self.global_best
        # 存储每次迭代的结果，画出收敛图
        self.iter_x = [0]
        self.iter_y = [init_l]

    def greedy_init(self, dis_mat, num_total, num_city):
        start_index = 0
        result = []
        for i in range(num_total):
            rest = [x for x in range(0, num_city)]
            # 所有起始点都已经生成了
            if start_index >= num_city:
                start_index = np.random.randint(0, num_city)
                result.append(result[start_index].copy())
                continue
            current = start_index
            rest.remove(current)
            # 找到一条最近邻路径
            result_one = [current]
            while len(rest) != 0:
                tmp_min = math.inf
                tmp_choose = -1
                for x in rest:
                    if dis_mat[current][x] < tmp_min:
                        tmp_min = dis_mat[current][x]
                        tmp_choose = x

                current = tmp_choose
                result_one.append(tmp_choose)
                rest.remove(tmp_choose)
            result.append(result_one)
            start_index += 1
        return result

    # 随机初始化
    def random_init(self, num_total, num_city):
        tmp = [x for x in range(num_city)]
        result = []
        for i in range(num_total):
            random.shuffle(tmp)
            result.append(tmp.copy())
        return result

    # 计算不同城市之间的距离
    def compute_dis_mat(self, num_city, location):
        dis_mat = np.zeros((num_city, num_city))
        for i in range(num_city):
            for j in range(num_city):
                if i == j:
                    dis_mat[i][j] = np.inf
                    continue
                a = location[i]
                b = location[j]
                tmp = np.sqrt(sum([(x[0] - x[1]) ** 2 for x in zip(a, b)]))
                dis_mat[i][j] = tmp
        return dis_mat

    # 计算一条路径的长度
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
            length = self.compute_pathlen(one, self.dis_mat)
            result.append(length)
        return result

    # 评估当前的群体
    def eval_particals(self):
        min_lenth = min(self.lenths)
        min_index = self.lenths.index(min_lenth)
        cur_path = self.particals[min_index]
        # 更新当前的全局最优
        if min_lenth < self.global_best_len:
            self.global_best_len = min_lenth
            self.global_best = cur_path
        # 更新当前的个体最优
        for i, l in enumerate(self.lenths):
            if l < self.local_best_len[i]:
                self.local_best_len[i] = l
                self.local_best[i] = self.particals[i]

    # 粒子交叉
    def cross(self, cur, best):
        one = cur.copy()
        l = [t for t in range(self.num_city)]
        t = np.random.choice(l, 2)
        x = min(t)
        y = max(t)
        cross_part = best[x:y]
        tmp = []
        for t in one:
            if t in cross_part:
                continue
            tmp.append(t)
        # 两种交叉方法
        one = tmp + cross_part
        l1 = self.compute_pathlen(one, self.dis_mat)
        one2 = cross_part + tmp
        l2 = self.compute_pathlen(one2, self.dis_mat)
        if l1 < l2:
            return one, l1
        else:
            return one, l2

    # 粒子变异
    def mutate(self, one):
        one = one.copy()
        l = [t for t in range(self.num_city)]
        t = np.random.choice(l, 2)
        x, y = min(t), max(t)
        one[x], one[y] = one[y], one[x]
        l2 = self.compute_pathlen(one, self.dis_mat)
        return one, l2

    # 迭代操作
    def pso(self):
        for cnt in range(1, self.iter_max):
            # 更新粒子群
            for i, one in enumerate(self.particals):
                tmp_l = self.lenths[i]
                # 与当前个体局部最优解进行交叉
                new_one, new_l = self.cross(one, self.local_best[i])
                if new_l < self.best_l:
                    self.best_l = tmp_l
                    self.best_path = one

                if new_l < tmp_l or np.random.rand() < 0.1:
                    one = new_one
                    tmp_l = new_l

                # 与当前全局最优解进行交叉
                new_one, new_l = self.cross(one, self.global_best)

                if new_l < self.best_l:
                    self.best_l = tmp_l
                    self.best_path = one

                if new_l < tmp_l or np.random.rand() < 0.1:
                    one = new_one
                    tmp_l = new_l
                # 变异
                one, tmp_l = self.mutate(one)

                if new_l < self.best_l:
                    self.best_l = tmp_l
                    self.best_path = one

                if new_l < tmp_l or np.random.rand() < 0.1:
                    one = new_one
                    tmp_l = new_l

                # 更新该粒子
                self.particals[i] = one
                self.lenths[i] = tmp_l
            # 评估粒子群，更新个体局部最优和个体当前全局最优
            self.eval_particals()
            # 更新输出解
            if self.global_best_len < self.best_l:
                self.best_l = self.global_best_len
                self.best_path = self.global_best
            self.iter_x.append(cnt)
            self.iter_y.append(self.best_l)
        return self.best_l, self.best_path

    def run(self):
        best_length, best_path = self.pso()
        best_path.append(best_path[0])
        # 画出最终路径
        return self.location[best_path], self.iter_y

