#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# %% Packages


from scipy.special import comb
import numpy as np
import pickle
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import tikzplotlib

sns.set(font='serif')
plt.rcParams.update({"text.usetex": True})
# plt.rcParams['font.family'] = 'serif'
# plt.rcParams['mathtext.fontset'] = 'dejavuserif'


# %% Functions


def level_position(level, dimension, state):
    """
    Calculates the position of *state* in *level*

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
        Position of state in level, or -1 if state is not in level.
    """

    level += dimension
    state = [clone + 1 for clone in state]

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


# %% Generating heatmaps


plotted_state = [4, 8, 8]
experiments = ['Hard', 'Soft']

label_size = 16
title_size = 18

for folder in experiments:
    fig, graphs = plt.subplots(2, 2, constrained_layout=True)
    matrices = 4
    for current_matrix in range(matrices):
        row = int(current_matrix / 2)
        col = current_matrix % 2
        division_distributions = [[] for _ in range(3)]
        for current_clone in range(3):
            file = open("{0}/Matrix-{1}/Clone-{2}/Parameters-{3}.bin".format(folder, current_matrix, current_clone + 1, current_matrix), 'rb')
            load_data = pickle.load(file)
            file.close()

            dimension_value = load_data[1]
            max_level_value = load_data[2]
            num_divisions = load_data[6]

            del load_data

            indexes = [i for i in range(num_divisions + 1)]

            for current_division in range(num_divisions + 1):
                file = open("{0}/Matrix-{1}/Clone-{2}/Data-{3}.bin".format(folder, current_matrix, current_clone + 1, current_division), 'rb')
                data = pickle.load(file)

                probability_value = data[sum(plotted_state)].todense()[level_position(sum(plotted_state), dimension_value, plotted_state)].tolist()[0][0]
                division_distributions[current_clone].append(probability_value)
                file.close()

            normalising_constant = sum(division_distributions[current_clone])
            # print(normalising_constant)
            for division in range(num_divisions + 1):
                division_distributions[current_clone][division] = division_distributions[current_clone][division] / normalising_constant

        # fig, graph = plt.subplots(1, 1)
        # graph.plot(indexes, division_distributions[0], color='black', linestyle='solid', label='$\mathcal{D}_{1}$')
        # graph.plot(indexes, division_distributions[1], color='red', linestyle='dashed', label='$\mathcal{D}_{2}$')
        # graph.plot(indexes, division_distributions[2], color='blue', linestyle='dotted', label='$\mathcal{D}_{3}$')
        # graph.legend(loc='best', facecolor='white', framealpha=1, fontsize=13)
        # graph.set_ylim(0, 0.4)
        # graph.set_xlim(0, 50)
        # graph.set_facecolor('white')
        # graph.spines['bottom'].set_color('gray')
        # graph.spines['top'].set_color('gray')
        # graph.spines['right'].set_color('gray')
        # graph.spines['left'].set_color('gray')
        ## graph.yaxis.grid(color='gray')
        ## graph.xaxis.grid(color='gray')

        graphs[row, col].plot(indexes, division_distributions[0], color='black', linestyle='solid', label='$\mathcal{D}_{1}$')
        graphs[row, col].plot(indexes, division_distributions[1], color='red', linestyle='dashed', label='$\mathcal{D}_{2}$')
        graphs[row, col].plot(indexes, division_distributions[2], color='blue', linestyle='dotted', label='$\mathcal{D}_{3}$')
        graphs[row, col].legend(loc='best', facecolor='white', framealpha=1, fontsize=label_size)
        # if folder == 'Hard':
        #     graphs[row, col].set_ylim(0, 0.2)
        # if folder == 'Soft':
        #     graphs[row, col].set_ylim(0, 0.3)
        if row == 1:
            graphs[row, col].set_xlabel('$\\textrm{Number of divisions}$', fontsize=label_size)
        graphs[row, col].set_ylim(0, 0.3)
        graphs[row, col].set_xlim(0, 35)
        graphs[row, col].set_facecolor('white')
        graphs[row, col].spines['bottom'].set_color('gray')
        graphs[row, col].spines['top'].set_color('gray')
        graphs[row, col].spines['right'].set_color('gray')
        graphs[row, col].spines['left'].set_color('gray')


        # plt.title('Distribution of divisions for $\\mathbf{n}_{0}=(' + '{0}, {1}, {2})$'.format(plotted_state[0], plotted_state[1], plotted_state[2]))

        # fig.savefig('DD-{0}-{1}-[{2},{3},{4}].pdf'.format(folder[0], current_matrix, plotted_state[0], plotted_state[1], plotted_state[2]))

        # tikzplotlib.save('{0}/Matrix{}/DD-{1}-C-{2}-{3}-{4}.tex'.format(folder, current_matrix, plotted_state[0], plotted_state[1], plotted_state[2]))
        # tikzplotlib.clean_figure()
        # tikzplotlib.save('DD-{0}-{1}-[{2},{3},{4}].tex'.format(folder[0], current_matrix, plotted_state[0], plotted_state[1], plotted_state[2]))

        # graph.clear()

    # fig.suptitle('$\\textrm{Distribution of divisions for the ' + folder.lower() + ' niche case}$')
    fig.suptitle('$\\textrm{' + folder + ' niche case}$', fontsize=title_size)

    fig.savefig('DD-{}.pdf'.format(folder[0]))

    for row in range(2):
        for col in range(2):
            graphs[row][col].clear()
    fig.clear()
    plt.close(fig='all')
