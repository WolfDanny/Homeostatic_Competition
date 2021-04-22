#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# %% Packages


from scipy.special import comb
import numpy as np
import pickle
import tikzplotlib
import seaborn as sns
import matplotlib.pyplot as plt

# sns.set(font='serif')
# plt.rcParams['font.family'] = 'serif'
# plt.rcParams['mathtext.fontset'] = 'dejavuserif'


# %% Functions


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


def level_states(level, dimension):
    """
    Creates a list of all non-absorbed states in *level*.

    Parameters
    ----------
    level : int
        Level of the state space.
    dimension : int
        Number of clonotypes.

    Returns
    -------
    state_list : List
        List of all states in level.
    """

    state_list = []
    n = [1 for _ in range(dimension)]

    while True:

        if len(n) == dimension and sum(n) == level and (n.count(0) == 0):
            state_list.append(n[:])

        n[0] += 1
        for i in range(len(n)):
            if n[i] > level - dimension + 1:
                if (i + 1) < len(n):
                    n[i+1] += 1
                    n[i] = 1
                for j in range(i):
                    n[j] = 1

        if n[-1] > level - dimension + 1:
            break

    return state_list


# %% Generating plots


plotted_state = [8, 8, 8]
experiments = ['Hard', 'Soft']
matrices = 4

for folder in experiments:
    for current_matrix in range(matrices):
        file = open("{0}/Parameters-{1}.bin".format(folder, current_matrix), 'rb')
        load_data = pickle.load(file)
        file.close()

        dimension_value = load_data[1]
        max_level_value = load_data[2]
        stimulus_value = load_data[5]

        del load_data

        file = open("{0}/Data-{1}.bin".format(folder, current_matrix), 'rb')
        distribution = pickle.load(file)
        file.close()

        absorption_distribution = []
        absorption_values = []
        level_lists = []
        # row = sum(plotted_state) - dimension_value
        # plotted_position = level_position(sum(plotted_state), dimension_value, plotted_state)

        indexes = [i + 1 for i in range(max_level_value - 2)]
        X, Y = np.meshgrid(indexes, indexes)

        absorption_c1_list = [distribution[0][i][sum(plotted_state) - dimension_value].todense().tolist()[level_position(sum(plotted_state), dimension_value, plotted_state)] for i in range(max_level_value - 2)]
        absorption_c1 = np.zeros((max_level_value - 2, max_level_value - 2))
        for current_level in range(len(distribution[0])):
            current_states = level_states(current_level + 2, 2)
            for current_state in current_states:
                absorption_c1[current_state[0] - 1, current_state[1] - 1] += absorption_c1_list[current_level][level_position(current_level + 2, 2, current_state)]
                # absorption_c1[current_state[0] - 1, current_state[1] - 1] += distribution[0][current_level][row].todense().tolist()[plotted_position][level_position(current_level + 2, 2, current_state)]

        absorption_distribution.append(absorption_c1)

        absorption_c2_list = [distribution[1][i][sum(plotted_state) - dimension_value].todense().tolist()[level_position(sum(plotted_state), dimension_value, plotted_state)] for i in range(max_level_value - 2)]
        absorption_c2 = np.zeros((max_level_value - 2, max_level_value - 2))
        for current_level in range(len(distribution[1])):
            current_states = level_states(current_level + 2, 2)
            for current_state in current_states:
                absorption_c2[current_state[0] - 1, current_state[1] - 1] += absorption_c2_list[current_level][level_position(current_level + 2, 2, current_state)]
                # absorption_c2[current_state[0] - 1, current_state[1] - 1] += distribution[1][current_level][row].todense().tolist()[plotted_position][level_position(current_level + 2, 2, current_state)]

        absorption_distribution.append(absorption_c2)

        absorption_c3_list = [distribution[2][i][sum(plotted_state) - dimension_value].todense().tolist()[level_position(sum(plotted_state), dimension_value, plotted_state)] for i in range(max_level_value - 2)]
        absorption_c3 = np.zeros((max_level_value - 2, max_level_value - 2))
        for current_level in range(len(distribution[2])):
            current_states = level_states(current_level + 2, 2)
            for current_state in current_states:
                absorption_c3[current_state[0] - 1, current_state[1] - 1] += absorption_c3_list[current_level][level_position(current_level + 2, 2, current_state)]
                # absorption_c3[current_state[0] - 1, current_state[1] - 1] += distribution[2][current_level][row].todense().tolist()[plotted_position][level_position(current_level + 2, 2, current_state)]

        absorption_distribution.append(absorption_c3)

        for i in range(len(absorption_distribution)):
            absorption_values.append(absorption_distribution[i].sum())

        # Conditional probabilities
        for i in range(len(absorption_distribution)):
            absorption_distribution[i] = absorption_distribution[i] / absorption_distribution[i].sum()

        for i in range(len(absorption_distribution)):
            raw_levels = np.linspace(absorption_distribution[i].min(), absorption_distribution[i].max(), 5000)
            level_values = [0.15, 0.35, 0.55, 0.75, 0.95]
            refined_levels = []
            for current_raw_level in range(raw_levels.shape[0]):
                total = 0.0
                for row in range(len(absorption_distribution[i])):
                    for col in range(len(absorption_distribution[i][row])):
                        if absorption_distribution[i][row][col] <= raw_levels[current_raw_level]:
                            total += absorption_distribution[i][row][col]
                if len(refined_levels) != len(level_values):
                    if level_values[len(refined_levels)] <= total:
                        refined_levels.append(raw_levels[current_raw_level])
            level_lists.append(refined_levels)

        fig, graph = plt.subplots(1, 1)
        CS = graph.contour(X, Y, absorption_distribution[0], level_lists[0], colors='black', linestyles='solid', alpha=1)
        CS1 = graph.contour(X, Y, absorption_distribution[1], level_lists[1], colors='red', linestyles='dashed', alpha=1)
        CS2 = graph.contour(X, Y, absorption_distribution[2], level_lists[2], colors='blue', linestyles='dotted', alpha=1)

        h1, _ = CS.legend_elements()
        h2, _ = CS1.legend_elements()
        h3, _ = CS2.legend_elements()
        graph.legend([h1[0], h2[0], h3[0]], ['$\mathbf{U}^{1}$', '$\mathbf{U}^{2}$', '$\mathbf{U}^{3}$'], facecolor='white', framealpha=1, fontsize=13)
        graph.set_facecolor('white')
        graph.spines['bottom'].set_color('gray')
        graph.spines['top'].set_color('gray')
        graph.spines['right'].set_color('gray')
        graph.spines['left'].set_color('gray')
        plt.title('Absorption distribution for $\\mathbf{n}_{0}=(' + '{0}, {1}, {2})$'.format(plotted_state[0], plotted_state[1], plotted_state[2]))

        fig.savefig('{0}/AD-{1}-C-[{2},{3},{4}].pdf'.format(folder, current_matrix, plotted_state[0], plotted_state[1], plotted_state[2]))
        tikzplotlib.save('{0}/AD-{1}-C-[{2},{3},{4}].tex'.format(folder, current_matrix, plotted_state[0], plotted_state[1], plotted_state[2]))

        graph.clear()
        fig.clear()
        plt.close(fig='all')

        absorption_values_plot = tuple([tuple([value, value, value]) for value in absorption_values])
        labels = ['Label 1', 'Label 2', 'Label 3']

        dim = len(absorption_values_plot[0])
        w = 0.75
        dimw = w / dim

        fig, graph = plt.subplots()
        x = np.arange(len(absorption_values_plot))
        for i in range(len(absorption_values_plot[0])):
            y = [d[i] for d in absorption_values_plot]
            b = graph.bar(x + i * dimw, y, dimw, bottom=0.001, label=labels[i])

        graph.set_xticks(x + dimw)
        graph.set_xticklabels(map(str, x))
        graph.legend(facecolor='white', framealpha=1, fontsize=13)
        graph.set_facecolor('white')
        graph.spines['bottom'].set_color('gray')
        graph.spines['top'].set_color('gray')
        graph.spines['right'].set_color('gray')
        graph.spines['left'].set_color('gray')

        plt.title('Probability of first extinction')

        fig.savefig('{0}/AD-{1}-C-[{2},{3},{4}]-FE.pdf'.format(folder, current_matrix, plotted_state[0], plotted_state[1], plotted_state[2]))
        tikzplotlib.save('{0}/AD-{1}-C-[{2},{3},{4}]-FE.tex'.format(folder, current_matrix, plotted_state[0], plotted_state[1], plotted_state[2]))

        graph.clear()
        fig.clear()
        plt.close(fig='all')
