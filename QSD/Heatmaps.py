#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#%% Packages
from scipy.special import comb
import numpy as np
import pickle
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D

sns.set(font='serif')

#%% Functions


def level_position(level, dimension, state):
    """
    Calculates the position of *state* in *level*.

    Parameters
    ----------
    level : int
        Level of the state space.
    dimension : int
        Number of clonotypes.
    state : List[int]
        List of number of cells per clonotype.

    Returns
    -------
    int
        Position of state in level, or -1 if state is not in the level.
    """

    if level == dimension and state.count(1) == dimension:
        return 0

    if len(state) != dimension or sum(state) != level or state.count(0) > 0:
        return -1

    position = 0

    max_cells = level - dimension + 1

    for i in range(dimension):
        position += (state[i] - 1) * (max_cells ** i)

    for i in range(dimension - 2):
        position += (state[dimension - 1 - i] - 1) * (1 - (max_cells ** (dimension - 1 - i)))

    position = int(position / (max_cells - 1))

    for i in range(dimension - 2):
        position += int(comb(level - 1 - sum(state[dimension - i:dimension]), dimension - 1 - i)) - int(comb(level - sum(state[dimension - i - 1:dimension]), dimension - 1 - i))

    return int(position - 1)

#%% Generating heatmaps


epsilon = 0.00001

max_levels = [[] for _ in range(2)]
max_values = [[] for _ in range(2)]

for i in range(2):
    for j in range(20):
        file = open("Model-{0}/QSD-{1}/Parameters.bin".format(i, j), 'rb')
        load_data = pickle.load(file)
        file.close()

        dimension_value = load_data[1]
        max_level_value = load_data[2]
        stimulus_value = load_data[5]

        del load_data

        file = open("Model-{0}/QSD-{1}/Data.bin".format(i, j), 'rb')
        data = pickle.load(file)
        file.close()

        captured = 0
        for captured_level in range(len(data)):
            captured += sum(data[captured_level])
            if captured >= 1 - epsilon:
                captured_level += 1
                break

        distribution = np.zeros((captured_level - dimension_value + 1, captured_level - dimension_value + 1, captured_level - dimension_value + 1))
        for k in range(len(distribution)):
            for m in range(len(distribution[k])):
                for n in range(len(distribution[k][m])):
                    if k + m + n <= captured_level:
                        distribution[k][m][n] += data[k + m + n][level_position(k + m + n + dimension_value, dimension_value, [k + 1, m + 1, n + 1])]

        max_levels[i].append(captured_level)

        max_values[i].append([[np.where(distribution == np.amax(distribution))[0][0], np.where(distribution == np.amax(distribution))[1][0], np.where(distribution == np.amax(distribution))[2][0]], distribution.max()])

        indexes = [i + 1 for i in range(captured_level - dimension_value + 1)]

        fig, graph = plt.subplots(1, 1)
        for k in range(len(distribution)):
            current_data = pd.DataFrame(distribution[k], columns=indexes, index=indexes)
            results_map = sns.heatmap(current_data, cmap="Blues", xticklabels=captured_level - dimension_value, yticklabels=captured_level - dimension_value, linewidths=.5, vmax=distribution.max(), vmin=0.0)
            results_map.invert_yaxis()
            plt.xlabel('$n_{3}$', fontsize=13)
            plt.ylabel('$n_{2}$', fontsize=13)
            plt.title('$n_{1}' + '={0}$, $\\varphi=({1},{2},{3})$, $\\varepsilon={4}$'.format(k + 1, stimulus_value[0], stimulus_value[1], stimulus_value[2], epsilon))
            fig.savefig('Model-{0}/QSD-{1}/QSD-n1-{2}.pdf'.format(i, j, k))

            results_map.clear()
            graph.clear()
            fig.clear()
        plt.close()

        # fig = plt.figure()
        # ax = fig.add_subplot(111, projection='3d')
        #
        # indexes = [i + 1 for i in range(captured_level - dimension_value + 1)]
        # x, y, z = np.meshgrid(indexes, indexes, indexes)
        #
        # img = ax.scatter(x, y, z, c=distribution, cmap=plt.hot())
        # fig.colorbar(img)
        # plt.show()

pickle.dump(max_levels, open('Truncated_levels.bin', 'wb'))
pickle.dump(max_values, open('Maximum_values.bin', 'wb'))
