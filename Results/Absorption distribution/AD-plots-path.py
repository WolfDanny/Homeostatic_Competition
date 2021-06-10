#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# %% Packages


from scipy.special import comb
from copy import deepcopy
import numpy as np
import pickle
import tikzplotlib
import matplotlib.pyplot as plt

plt.rcParams.update({"text.usetex": True})
plt.rcParams['font.family'] = 'serif'
plt.rcParams['mathtext.fontset'] = 'dejavuserif'

lw = 1.0


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

hard_data = []
soft_data = []

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

        if folder == 'Hard':
            hard_data.append(deepcopy(absorption_values_hard))
        else:
            soft_data.append(deepcopy(absorption_values_soft))

fig, graphs = plt.subplots(2, 2, figsize=(7, 7), constrained_layout=True)

graphs[0, 0].plot(absorption_index, hard_data[0][0], '-', lw=lw, color="b")
graphs[0, 0].plot(absorption_index, hard_data[1][0], '--', lw=lw, color="b")
graphs[0, 0].plot(absorption_index, hard_data[2][0], '-.', lw=lw, color="b")
graphs[0, 0].plot(absorption_index, hard_data[3][0], ':', lw=lw, color="b")
graphs[0, 0].plot(absorption_index, soft_data[0][0], '-', lw=lw, color="g")
graphs[0, 0].plot(absorption_index, soft_data[1][0], '--', lw=lw, color="g")
graphs[0, 0].plot(absorption_index, soft_data[2][0], '-.', lw=lw, color="g")
graphs[0, 0].plot(absorption_index, soft_data[3][0], ':', lw=lw, color="g")
graphs[0, 0].set_title('Probability of extinction of clonotype 1')
graphs[0, 0].set_xlabel('$n_{1}$')
graphs[0, 0].set_facecolor('white')
graphs[0, 0].set_ylim(0, 1)
graphs[0, 0].set_xlim(1, max_level_value - initial_state[0] - initial_state[1])
# graphs[0, 0].spines['bottom'].set_color('gray')
# graphs[0, 0].spines['top'].set_color('gray')
# graphs[0, 0].spines['right'].set_color('gray')
# graphs[0, 0].spines['left'].set_color('gray')

graphs[0, 1].plot(absorption_index, hard_data[0][1], '-', lw=lw, color="b")
graphs[0, 1].plot(absorption_index, hard_data[1][1], '--', lw=lw, color="b")
graphs[0, 1].plot(absorption_index, hard_data[2][1], '-.', lw=lw, color="b")
graphs[0, 1].plot(absorption_index, hard_data[3][1], ':', lw=lw, color="b")
graphs[0, 1].plot(absorption_index, soft_data[0][1], '-', lw=lw, color="g")
graphs[0, 1].plot(absorption_index, soft_data[1][1], '--', lw=lw, color="g")
graphs[0, 1].plot(absorption_index, soft_data[2][1], '-.', lw=lw, color="g")
graphs[0, 1].plot(absorption_index, soft_data[3][1], ':', lw=lw, color="g")
graphs[0, 1].set_title('Probability of extinction of clonotype 2')
graphs[0, 1].set_xlabel('$n_{1}$')
graphs[0, 1].set_facecolor('white')
graphs[0, 1].set_ylim(0, 1)
graphs[0, 1].set_xlim(1, max_level_value - initial_state[0] - initial_state[1])
# graphs[0, 1].spines['bottom'].set_color('gray')
# graphs[0, 1].spines['top'].set_color('gray')
# graphs[0, 1].spines['right'].set_color('gray')
# graphs[0, 1].spines['left'].set_color('gray')

graphs[1, 0].plot(absorption_index, hard_data[0][2], '-', lw=lw, color="b")
graphs[1, 0].plot(absorption_index, hard_data[1][2], '--', lw=lw, color="b")
graphs[1, 0].plot(absorption_index, hard_data[2][2], '-.', lw=lw, color="b")
graphs[1, 0].plot(absorption_index, hard_data[3][2], ':', lw=lw, color="b")
graphs[1, 0].plot(absorption_index, soft_data[0][2], '-', lw=lw, color="g")
graphs[1, 0].plot(absorption_index, soft_data[1][2], '--', lw=lw, color="g")
graphs[1, 0].plot(absorption_index, soft_data[2][2], '-.', lw=lw, color="g")
graphs[1, 0].plot(absorption_index, soft_data[3][2], ':', lw=lw, color="g")
graphs[1, 0].set_title('Probability of extinction of clonotype 3')
graphs[1, 0].set_xlabel('$n_{1}$')
graphs[1, 0].set_facecolor('white')
graphs[1, 0].set_ylim(0, 1)
graphs[1, 0].set_xlim(1, max_level_value - initial_state[0] - initial_state[1])
# graphs[1, 0].spines['bottom'].set_color('gray')
# graphs[1, 0].spines['top'].set_color('gray')
# graphs[1, 0].spines['right'].set_color('gray')
# graphs[1, 0].spines['left'].set_color('gray')

graphs[1, 1].axis('off')
graphs[1, 1].plot([], [], '-', color="k", label='$\\textrm{Scenario } (a)$')
graphs[1, 1].plot([], [], '--', color="k", label='$\\textrm{Scenario } (b)$')
graphs[1, 1].plot([], [], '-.', color="k", label='$\\textrm{Scenario } (c)$')
graphs[1, 1].plot([], [], ':', color="k", label='$\\textrm{Scenario } (d)$')
graphs[1, 1].plot([], [], 's', color="b", label='$\\textrm{Hard niche}$')
graphs[1, 1].plot([], [], 's', color="g", label='$\\textrm{Soft niche}$')

# graphs[1, 1].plot([], [], '-', color="k", label='Scenario $(a)$')
# graphs[1, 1].plot([], [], '--', color="k", label='Scenario $(b)$')
# graphs[1, 1].plot([], [], '-.', color="k", label='Scenario $(c)$')
# graphs[1, 1].plot([], [], ':', color="k", label='Scenario $(d)$')
# graphs[1, 1].plot([], [], 's', color="b", label='Hard niche')
# graphs[1, 1].plot([], [], 's', color="g", label='Soft niche')
graphs[1, 1].legend(loc='center', fontsize='xx-large')
# prop={'size': 25}

fig.savefig('First-extinction.pdf')

# tikzplotlib.clean_figure()
# tikzplotlib.save('First-extinction.tex')

for row in range(2):
    for col in range(2):
        graphs[row, col].clear()
fig.clear()
plt.close(fig='all')

fig, graphs = plt.subplots(1, 4, figsize=(12, 3), constrained_layout=True)

graphs[0].plot(absorption_index, hard_data[0][0], '-', lw=lw, color="b")
graphs[0].plot(absorption_index, hard_data[3][0], ':', lw=lw, color="b")
graphs[0].plot(absorption_index, soft_data[0][0], '-', lw=lw, color="g")
graphs[0].plot(absorption_index, soft_data[3][0], ':', lw=lw, color="g")
graphs[0].set_title('Probability of extinction of clonotype 1')
graphs[0].set_xlabel('$n_{1}$')
graphs[0].set_facecolor('white')
graphs[0].set_ylim(0, 1)
graphs[0].set_xlim(1, max_level_value - initial_state[0] - initial_state[1])

graphs[1].plot(absorption_index, hard_data[0][1], '-', lw=lw, color="b")
graphs[1].plot(absorption_index, hard_data[3][1], ':', lw=lw, color="b")
graphs[1].plot(absorption_index, soft_data[0][1], '-', lw=lw, color="g")
graphs[1].plot(absorption_index, soft_data[3][1], ':', lw=lw, color="g")
graphs[1].set_title('Probability of extinction of clonotype 2')
graphs[1].set_xlabel('$n_{1}$')
graphs[1].set_facecolor('white')
graphs[1].set_ylim(0, 1)
graphs[1].set_xlim(1, max_level_value - initial_state[0] - initial_state[1])

graphs[2].plot(absorption_index, hard_data[0][2], '-', lw=lw, color="b")
graphs[2].plot(absorption_index, hard_data[3][2], ':', lw=lw, color="b")
graphs[2].plot(absorption_index, soft_data[0][2], '-', lw=lw, color="g")
graphs[2].plot(absorption_index, soft_data[3][2], ':', lw=lw, color="g")
graphs[2].set_title('Probability of extinction of clonotype 3')
graphs[2].set_xlabel('$n_{1}$')
graphs[2].set_facecolor('white')
graphs[2].set_ylim(0, 1)
graphs[2].set_xlim(1, max_level_value - initial_state[0] - initial_state[1])

graphs[3].axis('off')
graphs[3].plot([], [], '-', color="k", label='$\\textrm{Symmetric}$')
graphs[3].plot([], [], ':', color="k", label='$\\textrm{Overlapping}$')
graphs[3].plot([], [], 's', color="b", label='$\\textrm{Hard niche}$')
graphs[3].plot([], [], 's', color="g", label='$\\textrm{Soft niche}$')
graphs[3].legend(loc='center', fontsize='xx-large')

fig.savefig('First-extinction-smb.pdf')

for col in range(4):
    graphs[col].clear()
fig.clear()
plt.close(fig='all')

