"""https://zhuanlan.zhihu.com/p/436453994
遗传算法的基本步骤如下：

1. 初始化：生成一个随机解的种群（population），每个解称为个体（individual）。
2. 评估：使用适应度函数（fitness function）评估每个个体的适应度，适应度越高，个体被选中的概率越大。
3. 选择（Selection）：根据适应度选择个体进行繁殖，适应度高的个体有更大的概率被选中。
4. 交叉（Crossover）：通过交叉操作生成新的个体，即在两个个体的某些位置上交换它们的部分特征。
5. 变异（Mutation）：以一定的概率对个体进行随机变异，以增加种群的多样性。
6. 迭代：重复上述过程，直到满足终止条件（如达到最大迭代次数或找到满足条件的解）。

"""

import random
import numpy as np
from itertools import (
    permutations,                   # 排列
    combinations,                   # 组合
    combinations_with_replacement   # 组合(包含自己)
)



def fitness_function(individual: np.ndarray) -> np.ndarray:
    """适应度函数，这里我们寻找标准差差最小的数据

    Args:
        individual (np.ndarray): [batch, n]

    Returns:
        np.ndarray: [batch] 算的适应度
    """
    # std = np.sqrt(np.square(individual - individual.mean(axis=-1, keepdims=True)).mean(axis=-1))
    std = individual.std(axis=-1)
    return std


def selection(population: np.ndarray, fitnesses: np.ndarray, selected_size: int) -> np.ndarray:
    """选择函数

    Args:
        population (np.ndarray): [batch, n]
        fitnesses (np.ndarray): [batch]
        selected_size (int): 选择数据长度

    Returns:
        np.ndarray: 选择的数据
    """
    # argsort是升序排列,因此选择最小的值
    sort_index = np.argsort(fitnesses, axis=0)
    sort_index
    return population[sort_index[:selected_size]]


def crossover(parent1: np.ndarray, parent2: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """交叉函数，这里使用单点交叉
    [1, 2, 3, 4, 5, 6] -> 交叉点为3 -> [1, 2, 3, d, e, f]
    [a, b, c, d, e, f] -> 交叉点为3 -> [a, b, c, 4, 5, 6]

    Args:
        parent1 (np.ndarray): [n] 父代1
        parent2 (np.ndarray): [n] 父代2

    Returns:
        tuple[np.ndarray, np.ndarray]: 两个子代
    """
    crossover_point = np.random.randint(1, len(parent1) - 1)
    return np.concatenate([parent1[:crossover_point], parent2[crossover_point:]]), \
        np.concatenate([parent2[:crossover_point], parent1[crossover_point:]])


def mutate(individual: np.ndarray) -> np.ndarray:
    """变异函数，这里简单地将变异点的值随机化

    Args:
        individual (np.ndarray): [n]

    Returns:
        np.ndarray: [n]
    """
    mutation_point = np.random.randint(0, len(individual) - 1)
    individual[mutation_point] = np.random.randint(-100, 100)
    return individual


def genetic_algorithm(population: np.ndarray, num_generations: int, mutation_rate: float) -> np.ndarray:
    """遗传算法主函数
        1. 评估：使用适应度函数（fitness function）评估每个个体的适应度，适应度越高，个体被选中的概率越大。
        2. 选择（Selection）：根据适应度选择个体进行繁殖，适应度高的个体有更大的概率被选中。
        3. 交叉（Crossover）：通过交叉操作生成新的个体，即在两个个体的某些位置上交换它们的部分特征。
        4. 变异（Mutation）：以一定的概率对个体进行随机变异，以增加种群的多样性。
        5. 迭代：重复上述过程，直到满足终止条件（如达到最大迭代次数或找到满足条件的解）。

    Args:
        population (np.ndarray): [batch, n]
        num_generations (int): 代数
        mutation_rate (float): 变异率

    Returns:
        np.ndarray: 最好的结果
    """
    population_size = population.shape[0]
    selected_size = population_size // 2

    for _ in range(num_generations):
        # 计算当前适应度
        fitnesses = fitness_function(population)
        # 选择适应度好的数据
        selected = selection(population, fitnesses, selected_size)

        # 组合id
        permutation = list(combinations(range(selected_size), 2))
        # 随机采样id,采样出 population_size//2 组,经过交叉变异后会生成 population_size 个个体
        permutation = random.sample(permutation, selected_size)

        # 交叉和变异
        # 注意,每次选择2个父母,变异出2个孩子
        children = []
        for i, j in permutation:
            parent1, parent2 = selected[i], selected[j]
            child1, child2 = crossover(parent1, parent2)
            children.append(mutate(child1) if np.random.uniform(0, 1) < mutation_rate else child1)
            children.append(mutate(child2) if np.random.uniform(0, 1) < mutation_rate else child2)

        population = np.array(children)

    return min(population, key=fitness_function)


if __name__ == "__main__":
    # 参数设置
    population_size = 20    # 种群大小
    num_generations = 100   # 迭代次数
    mutation_rate = 0.1     # 变异率

    population = np.random.randint(-100, 100, size=(population_size, 10))
    print(population)
    print(fitness_function(population))

    # 执行遗传算法
    best_individual = genetic_algorithm(population, num_generations, mutation_rate)
    print(f"最优个体是：{best_individual}, 适应度为：{fitness_function(best_individual)}")
