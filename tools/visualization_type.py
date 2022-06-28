from matplotlib import pyplot as plt
import math


# 画图（坐标；最优路径；最优路径路程；运行时长；是否显示坐标，默认不显示）
def draw_path(cities, best_path, best_length, total_time, label=0):
    ax = plt.subplot(111)
    ax.scatter(cities[:, 0], cities[:, 1], marker='o', facecolors='none', edgecolors='k', s=30)
    if label:
        for i, city in enumerate(cities):
            ax.text(city[0], city[1], str(i), fontsize=10, color="r", style="italic", weight="light",
                    verticalalignment='center', horizontalalignment='right', rotation=0)

    ax.plot(best_path[:, 0], best_path[:, 1], color='blue')
    ax.set_title("total time:{:.2f}s  total dis:{:.2f}".format(total_time, best_length))
    plt.show()


def draw_dis_list(best_dis_list):
    x = range(0, len(best_dis_list))
    ax = plt.subplot(111)
    ax.plot(x, best_dis_list)
    plt.show()


def draw_dis_lists(lists):
    col = 1
    row = len(lists)

    for i in range(len(lists)):
        x = range(0, len(lists[i]))
        plt.subplot(row, col, i + 1)
        plt.plot(x, lists[i])
    plt.show()
