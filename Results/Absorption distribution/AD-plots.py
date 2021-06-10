#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# %% Packages


from scipy.special import comb
from copy import deepcopy
import numpy as np
import pickle
import tikzplotlib
import seaborn as sns
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.pyplot as plt

# sns.set(font='serif')
plt.rcParams['text.latex.preamble'] = r"\usepackage{graphicx}"
plt.rcParams.update({"text.usetex": True})
plt.rcParams['font.family'] = 'serif'
plt.rcParams['mathtext.fontset'] = 'dejavuserif'


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


plotted_state = [4, 8, 8]
experiments = ['Hard', 'Soft']
matrices = 4

max_values = [np.array([]), np.array([])]
distributions = []
means = []
pie_values = []

for folder in experiments:
    folder_distributions = []
    folder_means = []
    folder_pie_values = []

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
        matrix_means = []

        indexes = [i + 1 for i in range(max_level_value - 2)]
        X, Y = np.meshgrid(indexes, indexes)

        absorption_c1_list = [distribution[0][i][sum(plotted_state) - dimension_value].todense().tolist()[level_position(sum(plotted_state), dimension_value, plotted_state)] for i in range(max_level_value - 2)]
        absorption_c1 = np.zeros((max_level_value - 2, max_level_value - 2))
        for current_level in range(len(distribution[0])):
            current_states = level_states(current_level + 2, 2)
            for current_state in current_states:
                absorption_c1[current_state[0] - 1, current_state[1] - 1] += absorption_c1_list[current_level][level_position(current_level + 2, 2, current_state)]
        absorption_distribution.append(absorption_c1)

        absorption_c2_list = [distribution[1][i][sum(plotted_state) - dimension_value].todense().tolist()[level_position(sum(plotted_state), dimension_value, plotted_state)] for i in range(max_level_value - 2)]
        absorption_c2 = np.zeros((max_level_value - 2, max_level_value - 2))
        for current_level in range(len(distribution[1])):
            current_states = level_states(current_level + 2, 2)
            for current_state in current_states:
                absorption_c2[current_state[0] - 1, current_state[1] - 1] += absorption_c2_list[current_level][level_position(current_level + 2, 2, current_state)]
        absorption_distribution.append(absorption_c2)

        absorption_c3_list = [distribution[2][i][sum(plotted_state) - dimension_value].todense().tolist()[level_position(sum(plotted_state), dimension_value, plotted_state)] for i in range(max_level_value - 2)]
        absorption_c3 = np.zeros((max_level_value - 2, max_level_value - 2))
        for current_level in range(len(distribution[2])):
            current_states = level_states(current_level + 2, 2)
            for current_state in current_states:
                absorption_c3[current_state[0] - 1, current_state[1] - 1] += absorption_c3_list[current_level][level_position(current_level + 2, 2, current_state)]
        absorption_distribution.append(absorption_c3)

        for i in range(len(absorption_distribution)):
            absorption_values.append(absorption_distribution[i].sum())

        # Conditional probabilities
        for i in range(len(absorption_distribution)):
            absorption_distribution[i] = absorption_distribution[i] / absorption_distribution[i].sum()

        max_value = max(absorption_distribution[0].max(), absorption_distribution[1].max(), absorption_distribution[2].max())
        if folder == 'Hard':
            max_values[0] = np.append(max_values[0], max_value)
        elif folder == 'Soft':
            max_values[1] = np.append(max_values[1], max_value)

        for clone in range(3):
            mean_value = [0, 0]
            for row in range(absorption_distribution[clone].shape[0]):
                for col in range(absorption_distribution[clone].shape[1]):
                    mean_value[0] += (row + 1) * absorption_distribution[clone][row, col]
                    mean_value[1] += (col + 1) * absorption_distribution[clone][row, col]
            matrix_means.append(deepcopy(mean_value))

        folder_distributions.append(deepcopy(absorption_distribution))
        folder_means.append(deepcopy(matrix_means))
        folder_pie_values.append(deepcopy(absorption_values))

    distributions.append(deepcopy(folder_distributions))
    means.append(deepcopy(folder_means))
    pie_values.append(deepcopy(folder_pie_values))

max_values = [max_values[0].max(), max_values[1].max()]
ticks = np.arange(-1, 15, 5)
ticks[0] = 0

h = 9
w = (26 / 14) * h

# Combined figure
fig, graphs = plt.subplots(4, 8, figsize=(w, h), constrained_layout=True)
h_maps = np.empty(graphs.shape, dtype=object)

for folder in experiments:
    for current_matrix in range(4):
        general_row = 2 * int(current_matrix / 2)
        general_col = 4 * (current_matrix % 2)
        if folder == 'Hard':
            colour_max = max_values[0]
            distribution = distributions[0][current_matrix]
            colour_map = 'Greens'
            folder_number = 0
        elif folder == 'Soft':
            colour_max = max_values[1]
            distribution = distributions[1][current_matrix]
            colour_map = 'Blues'
            folder_number = 1
            general_row += 1

        for clone in range(3):
            mean_value = means[folder_number][current_matrix][clone]
            row = general_row
            col = general_col + clone
            labels = [1, 2, 3]
            labels.pop(labels.index(clone + 1))

            h_maps[row, col] = graphs[row, col].imshow(distribution[clone], cmap=colour_map, interpolation='none', vmin=0, vmax=colour_max)

            if folder == 'Hard':
                graphs[row, col].plot(plotted_state[labels[1] - 1] - 1, plotted_state[labels[0] - 1] - 1, "^", color="blue", ms=4)
                graphs[row, col].plot(mean_value[1] - 1, mean_value[0] - 1, "d", color="blue", ms=4)
            elif folder == 'Soft':
                graphs[row, col].plot(plotted_state[labels[1] - 1] - 1, plotted_state[labels[0] - 1] - 1, "^", color="black", ms=4)
                graphs[row, col].plot(mean_value[1] - 1, mean_value[0] - 1, "d", color="black", ms=4)

            if current_matrix == 0 and clone == 0:
                if folder == 'Hard':
                    c_bar = fig.colorbar(h_maps[row, col], ax=graphs[:, :4], location='bottom')  # , shrink=1)
                elif folder == 'Soft':
                    c_bar = fig.colorbar(h_maps[row, col], ax=graphs[:, 4:], location='bottom')
                c_bar.outline.set_visible(False)

            graphs[row, col].set_facecolor('white')
            graphs[row, col].spines['bottom'].set_color('white')
            graphs[row, col].spines['top'].set_color('white')
            graphs[row, col].spines['right'].set_color('white')
            graphs[row, col].spines['left'].set_color('white')
            graphs[row, col].set_xlim(-0.5, 14.5)
            graphs[row, col].set_ylim(-0.5, 14.5)
            # graphs[row, col].set_xlim(-0.5, 34.5)
            # graphs[row, col].set_ylim(-0.5, 34.5)
            graphs[row, col].set_xticks(ticks)
            # graphs[row, col].set_xticklabels(ticks + 1)
            if row == 3:
                graphs[row, col].set_xticklabels(ticks + 1)
            else:
                graphs[row, col].set_xticklabels([])
            graphs[row, col].set_yticks(ticks)
            graphs[row, col].set_yticklabels(ticks + 1, rotation=90)
            if col == 0:
                graphs[row, col].set_yticklabels(ticks + 1, rotation=90)
            else:
                graphs[row, col].set_yticklabels([])
            if row == 0:
                graphs[row, col].set_title('$\mathbf{U}^'+'{}$'.format(clone + 1))
            if row == 3:
                graphs[row, col].set_xlabel('$n_{}$'.format(labels[1]), fontsize=12)
            graphs[row, col].set_ylabel('$n_{}$'.format(labels[0]), fontsize=12)

        graphs[general_row, general_col + 3].pie(pie_values[folder_number][current_matrix], labels=['$\sum\mathbf{U}^{1}$', '$\sum\mathbf{U}^{2}$', '$\sum\mathbf{U}^{3}$'], labeldistance=1.2, autopct='$%.1f\%%$', wedgeprops=dict(width=0.09), colors=['r', 'g', 'b'], startangle=30, pctdistance=0.63)
        graphs[general_row, general_col + 3].axis('equal')

fig.savefig('AD.pdf')
# tikzplotlib.save('AD.tex')

for row in range(4):
    for col in range(8):
        graphs[row][col].clear()
fig.clear()
plt.close(fig='all')


# Single-scenario figures
for current_matrix in range(4):
    fig, graphs = plt.subplots(2, 4, figsize=(w / 2, h / 2), constrained_layout=True)
    h_maps = np.empty(graphs.shape, dtype=object)
    for folder in experiments:
        general_row = 0  # 2 * int(current_matrix / 2)
        general_col = 0  # 4 * (current_matrix % 2)
        if folder == 'Hard':
            colour_max = max_values[0]
            distribution = distributions[0][current_matrix]
            colour_map = 'Greens'
            folder_number = 0
        elif folder == 'Soft':
            colour_max = max_values[1]
            distribution = distributions[1][current_matrix]
            colour_map = 'Blues'
            folder_number = 1
            general_row += 1

        for clone in range(3):
            mean_value = means[folder_number][current_matrix][clone]
            row = general_row
            col = general_col + clone
            labels = [1, 2, 3]
            labels.pop(labels.index(clone + 1))

            h_maps[row, col] = graphs[row, col].imshow(distribution[clone], cmap=colour_map, interpolation='none', vmin=0, vmax=colour_max)

            if folder == 'Hard':
                graphs[row, col].plot(plotted_state[labels[1] - 1] - 1, plotted_state[labels[0] - 1] - 1, "^", color="blue", ms=4)
                graphs[row, col].plot(mean_value[1] - 1, mean_value[0] - 1, "d", color="blue", ms=4)
            elif folder == 'Soft':
                graphs[row, col].plot(plotted_state[labels[1] - 1] - 1, plotted_state[labels[0] - 1] - 1, "^", color="black", ms=4)
                graphs[row, col].plot(mean_value[1] - 1, mean_value[0] - 1, "d", color="black", ms=4)

            if clone == 0:  # current_matrix == 0 and
                if folder == 'Hard':
                    c_bar = fig.colorbar(h_maps[row, col], ax=graphs[:, :2], location='bottom')  # , shrink=1)  , ax=graphs[:, :4]
                elif folder == 'Soft':
                    c_bar = fig.colorbar(h_maps[row, col], ax=graphs[:, 2:], location='bottom')
                c_bar.outline.set_visible(False)

            graphs[row, col].set_facecolor('white')
            graphs[row, col].spines['bottom'].set_color('white')
            graphs[row, col].spines['top'].set_color('white')
            graphs[row, col].spines['right'].set_color('white')
            graphs[row, col].spines['left'].set_color('white')
            graphs[row, col].set_xlim(-0.5, 14.5)
            graphs[row, col].set_ylim(-0.5, 14.5)
            # graphs[row, col].set_xlim(-0.5, 34.5)
            # graphs[row, col].set_ylim(-0.5, 34.5)
            graphs[row, col].set_xticks(ticks)
            # graphs[row, col].set_xticklabels(ticks + 1)
            if row == 3:
                graphs[row, col].set_xticklabels(ticks + 1)
            else:
                graphs[row, col].set_xticklabels([])
            graphs[row, col].set_yticks(ticks)
            graphs[row, col].set_yticklabels(ticks + 1, rotation=90)
            if col == 0:
                graphs[row, col].set_yticklabels(ticks + 1, rotation=90)
            else:
                graphs[row, col].set_yticklabels([])
            if row == 0:
                graphs[row, col].set_title('$\mathbf{U}^'+'{}$'.format(clone + 1))
            if row == 1:  # 0
                graphs[row, col].set_xlabel('$n_{}$'.format(labels[1]), fontsize=12)
            graphs[row, col].set_ylabel('$n_{}$'.format(labels[0]), fontsize=12)

        graphs[general_row, general_col + 3].pie(pie_values[folder_number][current_matrix], labels=['$\sum\mathbf{U}^{1}$', '$\sum\mathbf{U}^{2}$', '$\sum\mathbf{U}^{3}$'], autopct='$%.1f\%%$', wedgeprops=dict(width=0.09), colors=['r', 'g', 'b'], startangle=30, pctdistance=0.63)
        graphs[general_row, general_col + 3].axis('equal')

    fig.savefig('AD-{}.pdf'.format(current_matrix + 1))
    for row in range(2):
        for col in range(4):
            graphs[row][col].clear()
    fig.clear()

plt.close(fig='all')

# h_map0 = graphs[0].imshow(absorption_distribution[0], cmap="Greens", interpolation='none', vmin=0, vmax=colour_max)
# graphs[0].set_facecolor('white')
# graphs[0].spines['bottom'].set_color('white')
# graphs[0].spines['top'].set_color('white')
# graphs[0].spines['right'].set_color('white')
# graphs[0].spines['left'].set_color('white')
# graphs[0].set_xlabel('$n_{2}$')
# graphs[0].set_ylabel('$n_{3}$')
# graphs[0].set_xlim(-0.5, 14.5)
# graphs[0].set_ylim(-0.5, 14.5)
# graphs[0].set_xticks(ticks)
# graphs[0].set_xticklabels(ticks + 1)
# graphs[0].set_yticks(ticks)
# graphs[0].set_yticklabels(ticks + 1, rotation=90)
# graphs[0].set_title('Clonotype 1')
#
# h_map1 = graphs[1].imshow(absorption_distribution[1], cmap="Greens", interpolation='none', vmin=0, vmax=colour_max)
# graphs[1].set_facecolor('white')
# graphs[1].spines['bottom'].set_color('white')
# graphs[1].spines['top'].set_color('white')
# graphs[1].spines['right'].set_color('white')
# graphs[1].spines['left'].set_color('white')
# graphs[1].set_xlabel('$n_{1}$')
# graphs[1].set_ylabel('$n_{3}$')
# graphs[1].set_xlim(-0.5, 14.5)
# graphs[1].set_ylim(-0.5, 14.5)
# graphs[1].set_xticks(ticks)
# graphs[1].set_xticklabels(ticks + 1)
# graphs[1].set_yticks(ticks)
# graphs[1].set_yticklabels(ticks + 1, rotation=90)
# graphs[1].set_title('Clonotype 2')
#
# h_map2 = graphs[2].imshow(absorption_distribution[2], cmap="Greens", interpolation='none', vmin=0, vmax=colour_max)
# graphs[2].set_facecolor('white')
# c_bar = fig.colorbar(h_map2, ax=graphs[:])
# c_bar.outline.set_visible(False)
# graphs[2].spines['bottom'].set_color('white')
# graphs[2].spines['top'].set_color('white')
# graphs[2].spines['right'].set_color('white')
# graphs[2].spines['left'].set_color('white')
# graphs[2].set_xlabel('$n_{1}$')
# graphs[2].set_ylabel('$n_{2}$')
# graphs[2].set_xlim(-0.5, 14.5)
# graphs[2].set_ylim(-0.5, 14.5)
# graphs[2].set_xticks(ticks)
# graphs[2].set_xticklabels(ticks + 1)
# graphs[2].set_yticks(ticks)
# graphs[2].set_yticklabels(ticks + 1, rotation=90)
# graphs[2].set_title('Clonotype 3')

# fig.savefig('AD-{0}-{1}.pdf'.format(current_matrix, folder[0]))
# # tikzplotlib.save('{0}/AD-{1}.tex'.format(folder, current_matrix))
#
# for col in range(3):
#     graphs[col].clear()
# fig.clear()
# plt.close(fig='all')

#%% Combined plot

        # h_map0 = graphs[0].imshow(absorption_distribution[0], cmap="Greens", interpolation='none', vmin=0, vmax=colour_max)
        # graphs[0].set_facecolor('white')
        # graphs[0].spines['bottom'].set_color('white')
        # graphs[0].spines['top'].set_color('white')
        # graphs[0].spines['right'].set_color('white')
        # graphs[0].spines['left'].set_color('white')
        # graphs[0].set_xlabel('$n_{2}$')
        # graphs[0].set_ylabel('$n_{3}$')
        # graphs[0].set_xticks(ticks)
        # graphs[0].set_xticklabels(ticks + 1)
        # graphs[0].set_yticks(ticks)
        # graphs[0].set_yticklabels(ticks + 1, rotation=90)
        # graphs[0].invert_yaxis()
        # graphs[0].set_title('$\mathbf{U}^{1} =' + '{:.3f}$'.format(absorption_values[0]))
        #
        # h_map1 = graphs[1].imshow(absorption_distribution[1], cmap="Greens", interpolation='none', vmin=0, vmax=colour_max)
        # graphs[1].set_facecolor('white')
        # graphs[1].spines['bottom'].set_color('white')
        # graphs[1].spines['top'].set_color('white')
        # graphs[1].spines['right'].set_color('white')
        # graphs[1].spines['left'].set_color('white')
        # graphs[1].set_xlabel('$n_{1}$')
        # graphs[1].set_ylabel('$n_{3}$')
        # graphs[1].set_xticks(ticks)
        # graphs[1].set_xticklabels(ticks + 1)
        # graphs[1].set_yticks(ticks)
        # graphs[1].set_yticklabels(ticks + 1, rotation=90)
        # graphs[1].invert_yaxis()
        # graphs[1].set_title('$\mathbf{U}^{2} =' + '{:.3f}$'.format(absorption_values[1]))
        #
        # h_map2 = graphs[2].imshow(absorption_distribution[2], cmap="Greens", interpolation='none', vmin=0, vmax=colour_max)
        # graphs[2].set_facecolor('white')
        # graphs[2].spines['bottom'].set_color('white')
        # graphs[2].spines['top'].set_color('white')
        # graphs[2].spines['right'].set_color('white')
        # graphs[2].spines['left'].set_color('white')
        # graphs[2].set_xlabel('$n_{1}$')
        # graphs[2].set_ylabel('$n_{2}$')
        # graphs[2].set_xticks(ticks)
        # graphs[2].set_xticklabels(ticks + 1)
        # graphs[2].set_yticks(ticks)
        # graphs[2].set_yticklabels(ticks + 1, rotation=90)
        # graphs[2].invert_yaxis()
        # graphs[2].set_title('$\mathbf{U}^{3} =' + '{:.3f}$'.format(absorption_values[2]))

        # c_bar = fig.colorbar(h_map0, ax=graphs[2])
        # c_bar.outline.set_visible(False)

        # fig.savefig('{0}/AD-{1}-C-[{2},{3},{4}].pdf'.format(folder, current_matrix, plotted_state[0], plotted_state[1], plotted_state[2]))
        # # tikzplotlib.save('{0}/AD-{1}-C-[{2},{3},{4}].tex'.format(folder, current_matrix, plotted_state[0], plotted_state[1], plotted_state[2]))
        #
        # graphs.clear()
        # fig.clear()
        # plt.close(fig='all')

        # for i in range(len(absorption_distribution)):
        #     raw_levels = np.linspace(absorption_distribution[i].min(), absorption_distribution[i].max(), 5000)
        #     level_values = [0.15, 0.35, 0.55, 0.75, 0.95]
        #     refined_levels = []
        #     for current_raw_level in range(raw_levels.shape[0]):
        #         total = 0.0
        #         for row in range(len(absorption_distribution[i])):
        #             for col in range(len(absorption_distribution[i][row])):
        #                 if absorption_distribution[i][row][col] <= raw_levels[current_raw_level]:
        #                     total += absorption_distribution[i][row][col]
        #         if len(refined_levels) != len(level_values):
        #             if level_values[len(refined_levels)] <= total:
        #                 refined_levels.append(raw_levels[current_raw_level])
        #     level_lists.append(refined_levels)

        # fig, graph = plt.subplots(1, 1)
        # CS = graph.contour(X, Y, absorption_distribution[0], level_lists[0], colors='black', linestyles='solid', alpha=1)
        # CS1 = graph.contour(X, Y, absorption_distribution[1], level_lists[1], colors='red', linestyles='dashed', alpha=1)
        # CS2 = graph.contour(X, Y, absorption_distribution[2], level_lists[2], colors='blue', linestyles='dotted', alpha=1)

        # h1, _ = CS.legend_elements()
        # h2, _ = CS1.legend_elements()
        # h3, _ = CS2.legend_elements()
        # graph.legend([h1[0], h2[0], h3[0]], ['$\mathbf{U}^{1}$', '$\mathbf{U}^{2}$', '$\mathbf{U}^{3}$'], facecolor='white', framealpha=1, fontsize=13)
        # graph.set_facecolor('white')
        # graph.spines['bottom'].set_color('gray')
        # graph.spines['top'].set_color('gray')
        # graph.spines['right'].set_color('gray')
        # graph.spines['left'].set_color('gray')
        # plt.title('Absorption distribution for $\\mathbf{n}_{0}=(' + '{0}, {1}, {2})$'.format(plotted_state[0], plotted_state[1], plotted_state[2]))

        # fig.savefig('{0}/AD-{1}-C-[{2},{3},{4}].pdf'.format(folder, current_matrix, plotted_state[0], plotted_state[1], plotted_state[2]))
        # tikzplotlib.save('{0}/AD-{1}-C-[{2},{3},{4}].tex'.format(folder, current_matrix, plotted_state[0], plotted_state[1], plotted_state[2]))

        # graph.clear()
        # fig.clear()
        # plt.close(fig='all')
        #
        # absorption_values_plot = tuple([tuple([value, value, value]) for value in absorption_values])
        # labels = ['Label 1', 'Label 2', 'Label 3']
        #
        # dim = len(absorption_values_plot[0])
        # w = 0.75
        # dimw = w / dim
        #
        # fig, graph = plt.subplots()
        # x = np.arange(len(absorption_values_plot))
        # for i in range(len(absorption_values_plot[0])):
        #     y = [d[i] for d in absorption_values_plot]
        #     b = graph.bar(x + i * dimw, y, dimw, bottom=0.001, label=labels[i])
        #
        # graph.set_xticks(x + dimw)
        # graph.set_xticklabels(map(str, x))
        # graph.legend(facecolor='white', framealpha=1, fontsize=13)
        # graph.set_facecolor('white')
        # graph.spines['bottom'].set_color('gray')
        # graph.spines['top'].set_color('gray')
        # graph.spines['right'].set_color('gray')
        # graph.spines['left'].set_color('gray')
        #
        # plt.title('Probability of first extinction')
        #
        # fig.savefig('{0}/AD-{1}-C-[{2},{3},{4}]-FE.pdf'.format(folder, current_matrix, plotted_state[0], plotted_state[1], plotted_state[2]))
        # tikzplotlib.save('{0}/AD-{1}-C-[{2},{3},{4}]-FE.tex'.format(folder, current_matrix, plotted_state[0], plotted_state[1], plotted_state[2]))
        #
        # graph.clear()
        # fig.clear()
        # plt.close(fig='all')
