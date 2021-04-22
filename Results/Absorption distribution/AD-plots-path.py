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


initial_state = [8, 8]
experiments = ['Hard', 'Soft']
matrices = 4

for current_matrix in range(matrices):

    absorption_values_hard = [[], [], []]
    absorption_values_soft = [[], [], []]

    for folder in experiments:
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

        absorption_index = [i + 1 for i in range(max_level_value - initial_state[0] - initial_state[1])]

        for i in range(max_level_value - initial_state[0] - initial_state[1]):
            plotted_state = [i + 1, initial_state[0], initial_state[1]]
            absorption_c1 = [distribution[0][i][sum(plotted_state) - dimension_value].todense().tolist()[level_position(sum(plotted_state), dimension_value, plotted_state)] for i in range(max_level_value - 2)]
            absorption_c2 = [distribution[1][i][sum(plotted_state) - dimension_value].todense().tolist()[level_position(sum(plotted_state), dimension_value, plotted_state)] for i in range(max_level_value - 2)]
            absorption_c3 = [distribution[2][i][sum(plotted_state) - dimension_value].todense().tolist()[level_position(sum(plotted_state), dimension_value, plotted_state)] for i in range(max_level_value - 2)]
            absorption_c1_value = sum([sum(current_level) for current_level in absorption_c1])
            absorption_c2_value = sum([sum(current_level) for current_level in absorption_c2])
            absorption_c3_value = sum([sum(current_level) for current_level in absorption_c3])

            if folder == 'Hard':
                absorption_values_hard[0].append(absorption_c1_value)
                absorption_values_hard[1].append(absorption_c2_value)
                absorption_values_hard[2].append(absorption_c3_value)
            else:
                absorption_values_soft[0].append(absorption_c1_value)
                absorption_values_soft[1].append(absorption_c2_value)
                absorption_values_soft[2].append(absorption_c3_value)

    fig, graph = plt.subplots()
    graph.plot(absorption_index, absorption_values_hard[0], '--', color="b", label='Clonotype 1 H')
    graph.plot(absorption_index, absorption_values_hard[1], '-', color="b", label='Clonotype 2 H ')
    graph.plot(absorption_index, absorption_values_hard[2], '.', color="b", label='Clonotype 3 H ')
    graph.plot(absorption_index, absorption_values_soft[0], '--', color="g", label='Clonotype 1 S')
    graph.plot(absorption_index, absorption_values_soft[1], '-', color="g", label='Clonotype 2 S')
    graph.plot(absorption_index, absorption_values_soft[2], '.', color="g", label='Clonotype 3 S')
    graph.set_ylabel('Probability of extinction')
    graph.set_xlabel('$n_{1}$')
    graph.set_facecolor('white')
    graph.legend(loc='upper right')
    graph.set_ylim(0, 1)
    graph.set_xlim(1, max_level_value - initial_state[0] - initial_state[1])
    graph.spines['bottom'].set_color('gray')
    graph.spines['top'].set_color('gray')
    graph.spines['right'].set_color('gray')
    graph.spines['left'].set_color('gray')

    fig.savefig('AL-{0}-[-,{1},{2}]-FE.pdf'.format(current_matrix, initial_state[0], initial_state[1]))
    tikzplotlib.clean_figure()
    tikzplotlib.save('AL-{0}-[-,{1},{2}]-FE.tex'.format(current_matrix, initial_state[0], initial_state[1]))

    graph.clear()
    fig.clear()
    plt.close(fig='all')
