# %% Packages


from scipy.special import comb
from copy import deepcopy
import numpy as np
import pickle
import matplotlib.pyplot as plt
from skimage import color, io

# from scipy.special import comb
# from copy import deepcopy
# import numpy as np
# import matplotlib.pyplot as plt
# import pickle
# import sys
# from skimage import color, io
# sys.path.append('../../Definitions/')
# from homeostatic import *

plt.rcParams.update({"text.usetex": True})
plt.rcParams['text.latex.preamble'] = r"\usepackage{graphicx}"
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


# %% Loading data


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
w = (38 / 22) * h

# %% Generating figure


wspacing = 0.05
hspacing = 1.5 * wspacing
scenario_names = ['$(a)$', '$(b)$', '$(c)$', '$(d)$']
pie_size = 10 + 2
tick_size = 12 + 2
label_size = 16 + 4
title_size = 18 + 4

fig = plt.figure(constrained_layout=True, figsize=(w, h))
LRfigs = fig.subfigures(2, 2, height_ratios=[12, 1], wspace=wspacing)
subfigs = np.array([LRfigs[0, 0].subfigures(2, 1, hspace=hspacing), LRfigs[0, 1].subfigures(2, 1, hspace=hspacing)])
subfig_list = np.empty(4, dtype=object)
piefig_list = np.empty(4, dtype=object)
subfig_h_maps = np.empty(2, dtype=object)

for current_matrix in range(4):
    fig_row = int((2 * int(current_matrix / 2)) / 2)
    fig_col = int((4 * (current_matrix % 2)) / 4)
    subfigs[fig_col][fig_row].suptitle(scenario_names[current_matrix], fontsize=title_size)
    plotfigs, piefigs = subfigs[fig_col][fig_row].subfigures(1, 2, width_ratios=[20, 9])
    subfig_list[current_matrix] = plotfigs.subplots(2, 3)
    piefig_list[current_matrix] = piefigs.subplots(2, 1)

    for folder in experiments:
        subfig_row = 0
        if folder == 'Hard':
            colour_max = max_values[0]
            distribution = distributions[0][current_matrix]
            colour_map = 'Greens'
            folder_number = 0
        elif folder == 'Soft':
            subfig_row += 1
            colour_max = max_values[1]
            distribution = distributions[1][current_matrix]
            colour_map = 'Blues'
            folder_number = 1
        subfig_col = 0

        for clone in range(3):
            mean_value = means[folder_number][current_matrix][clone]
            row = subfig_row
            col = subfig_col + clone
            labels = [1, 2, 3]
            labels.pop(labels.index(clone + 1))

            if fig_col == fig_row == folder_number == clone == 0:
                subfig_h_maps[0] = subfig_list[current_matrix][row, col].imshow(distribution[clone], cmap=colour_map, interpolation='none', vmin=0, vmax=colour_max)
            elif fig_col == fig_row == clone == 0 and folder_number == 1:
                subfig_h_maps[1] = subfig_list[current_matrix][row, col].imshow(distribution[clone], cmap=colour_map, interpolation='none', vmin=0, vmax=colour_max)
            else:
                subfig_list[current_matrix][row, col].imshow(distribution[clone], cmap=colour_map, interpolation='none', vmin=0, vmax=colour_max)

            if folder_number == 0:
                subfig_list[current_matrix][row, col].plot(plotted_state[labels[1] - 1] - 1, plotted_state[labels[0] - 1] - 1, "^", color="blue", ms=4)
                subfig_list[current_matrix][row, col].plot(mean_value[1] - 1, mean_value[0] - 1, "d", color="blue", ms=4)
            elif folder_number == 1:
                subfig_list[current_matrix][row, col].plot(plotted_state[labels[1] - 1] - 1, plotted_state[labels[0] - 1] - 1, "^", color="black", ms=4)
                subfig_list[current_matrix][row, col].plot(mean_value[1] - 1, mean_value[0] - 1, "d", color="black", ms=4)

            subfig_list[current_matrix][row, col].set_facecolor('white')
            subfig_list[current_matrix][row, col].spines['bottom'].set_color('white')
            subfig_list[current_matrix][row, col].spines['top'].set_color('white')
            subfig_list[current_matrix][row, col].spines['right'].set_color('white')
            subfig_list[current_matrix][row, col].spines['left'].set_color('white')
            subfig_list[current_matrix][row, col].set_xlim(-0.5, 14.5)
            subfig_list[current_matrix][row, col].set_ylim(-0.5, 14.5)
            subfig_list[current_matrix][row, col].set_xticks(ticks)

            if fig_row == row == 1:
                subfig_list[current_matrix][row, col].set_xticklabels(ticks + 1, fontsize=tick_size)
                subfig_list[current_matrix][row, col].set_xlabel('$n_{}$'.format(labels[1]), fontsize=label_size)
            else:
                subfig_list[current_matrix][row, col].set_xticklabels([])
            subfig_list[current_matrix][row, col].set_ylabel('$n_{}$'.format(labels[0]), fontsize=label_size)

            subfig_list[current_matrix][row, col].set_yticks(ticks)
            if fig_col == col == 0:
                subfig_list[current_matrix][row, col].set_yticklabels(ticks + 1, fontsize=tick_size, rotation=90)
            else:
                subfig_list[current_matrix][row, col].set_yticklabels([])

            if fig_row == row == 0:
                subfig_list[current_matrix][row, col].set_title('$\mathbf{U}^'+'{}$'.format(clone + 1), fontsize=title_size)

        # patches, text, autotext = piefig_list[current_matrix][row].pie(pie_values[folder_number][current_matrix], labels=['$\sum\mathbf{U}^{1}$', '$\sum\mathbf{U}^{2}$', '$\sum\mathbf{U}^{3}$'], labeldistance=1.2, autopct='$%.1f\%%$', wedgeprops=dict(width=0.08), colors=['r', 'g', 'b'], startangle=30, pctdistance=0.61, radius=1)

        # COLOUR TEST
        patches, text, autotext = piefig_list[current_matrix][row].pie(pie_values[folder_number][current_matrix], labels=['$\sum\mathbf{U}^{1}$', '$\sum\mathbf{U}^{2}$', '$\sum\mathbf{U}^{3}$'], labeldistance=1.2, autopct='$%.1f\%%$', wedgeprops=dict(width=0.08), colors=['#B3784B', '#78FF78', '#A591FF'], startangle=30, pctdistance=0.61-0.11, radius=1)
        # COLOUR TEST
        if fig_row == row == 1:
            piefig_list[current_matrix][row].set_xlabel('$n_{i}$', fontsize=label_size, color='w')
            piefig_list[current_matrix][row].set_xticks([0])
            piefig_list[current_matrix][row].set_xticklabels([0], fontsize=tick_size, color='w')
            piefig_list[current_matrix][row].tick_params(colors='w', which='both')
        for name in text:
            name.set_fontsize(tick_size)
        for name in autotext:
            name.set_fontsize(pie_size)
        piefig_list[current_matrix][row].axis('equal')

left_axis = LRfigs[1, 0].subplots(1)
left_axis.axis('off')
c_bar_l = fig.colorbar(subfig_h_maps[0], ax=left_axis, orientation='horizontal', fraction=0.975, aspect=40)
c_bar_l.ax.tick_params(labelsize=tick_size)
c_bar_l.outline.set_visible(False)

right_axis = LRfigs[1, 1].subplots(1)
right_axis.axis('off')
c_bar_r = fig.colorbar(subfig_h_maps[1], ax=right_axis, orientation='horizontal', fraction=0.975, aspect=40)
c_bar_r.ax.tick_params(labelsize=tick_size)
c_bar_r.outline.set_visible(False)

fig.savefig('AD.pdf')

# Colour tests
fig.savefig("AD.png", dpi=300)
test = io.imread("AD.png")
test = color.rgb2gray(color.rgba2rgb(test))
io.imsave("AD-G.png", test)
# Colour tests end

# fig.clear()  # CONFLICTS WITH SKIMAGE
plt.close(fig='all')
