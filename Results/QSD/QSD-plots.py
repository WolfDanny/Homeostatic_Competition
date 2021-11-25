# %% Packages


import pickle
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.special import comb
from copy import deepcopy

sns.set(font='serif')
sns.set_style("ticks")
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


# %% Plotting distributions


experiments = ['Hard', 'Soft']
epsilon = 0.0001

with open("Established/Means.bin", 'rb') as file:
    original_means = pickle.load(file)

max_values = [[np.array([]), np.array([])], [np.array([]), np.array([])]]
max_levels = []
marginals = []
histograms = []
means = []

for folder in experiments:
    folder_max_levels = []
    folder_marginals = []
    folder_histograms = []
    folder_means = []
    folder_number = 0
    if folder == 'Soft':
        folder_number += 1

    for current_model in range(2):
        model_max_levels = []
        model_marginals = []
        model_histograms = []
        model_means = []

        for current_matrix in range(4):
            try:
                file = open("{0}/Model-{1}/QSD-{2}-{3}/Parameters.bin".format(folder, current_model + 1, folder[0], current_matrix), 'rb')
                load_data = pickle.load(file)
                file.close()

                dimension_value = load_data[1]
                max_level_value = load_data[2]
                stimulus_value = load_data[5]

                del load_data

                file = open("{0}/Model-{1}/QSD-{2}-{3}/Data-{4}-{5}.bin".format(folder, current_model + 1, folder[0], current_matrix, current_model + 1, current_matrix), 'rb')
                data = pickle.load(file)
                file.close()

                captured = 0
                captured_level = 0  # So that it is not undefined

                for captured_level in range(len(data)):
                    captured += sum(data[captured_level])
                    if captured >= 1 - epsilon:
                        captured_level += 1
                        break
                model_max_levels.append(captured_level)

                captured_range = captured_level - dimension_value + 1

                mean_value = [0, 0, 0]
                distribution = np.zeros((captured_range, captured_range, captured_range))
                for k in range(len(distribution)):
                    for m in range(len(distribution[k])):
                        for n in range(len(distribution[k][m])):
                            if k + m + n <= captured_level:
                                distribution[k][m][n] += data[k + m + n][level_position(k + m + n + dimension_value, dimension_value, [k + 1, m + 1, n + 1])]
                                mean_value[0] += (k + 1) * distribution[k][m][n]
                                mean_value[1] += (m + 1) * distribution[k][m][n]
                                mean_value[2] += (n + 1) * distribution[k][m][n]
                model_means.append(deepcopy(mean_value))

                marginal_distribution = np.zeros((captured_range, captured_range))
                n1_distribution = np.zeros(captured_range)
                for d in range(captured_range):
                    for m in range(captured_range):
                        for n in range(captured_range):
                            marginal_distribution[d][m] += distribution[n][m][d]
                            n1_distribution[d] += distribution[d][n][m]

                if folder == 'Soft' and current_matrix == 3:
                    max_values[current_model][0] = np.append(max_values[current_model][0], 0)
                elif folder == 'Hard' and current_matrix == 3 and current_model == 0:
                    max_values[current_model][0] = np.append(max_values[current_model][0], 0)
                else:
                    max_values[current_model][0] = np.append(max_values[current_model][0], marginal_distribution.max())
                max_values[current_model][1] = np.append(max_values[current_model][1], n1_distribution.max())
                model_marginals.append(deepcopy(marginal_distribution))
                model_histograms.append(deepcopy(n1_distribution))

            except FileNotFoundError:
                max_values[current_model][0] = np.append(max_values[current_model][0], 0)
                model_max_levels.append(0)
                model_marginals.append([])
                model_histograms.append([])
                model_means.append([])

        folder_max_levels.append(deepcopy(model_max_levels))
        folder_marginals.append(deepcopy(model_marginals))
        folder_histograms.append(deepcopy(model_histograms))
        folder_means.append(deepcopy(model_means))

    max_levels.append(deepcopy(folder_max_levels))
    marginals.append(deepcopy(folder_marginals))
    histograms.append(deepcopy(folder_histograms))
    means.append(deepcopy(folder_means))


marginal_colour_max = [max_values[0][0].max(), max_values[1][0].max()]

ticks = np.arange(-1, 15, 5)
ticks[0] = 0

for current_model in range(2):
    fig, graphs = plt.subplots(4, 4, figsize=(16, 16), constrained_layout=True)
    for current_matrix in range(4):
        for folder in experiments:
            if folder == 'Hard':
                captured_level = max_levels[0][current_model][current_matrix]
                marginal_distribution = marginals[0][current_model][current_matrix]
                n1_distribution = histograms[0][current_model][current_matrix]
                mean_value = means[0][current_model][current_matrix]
                row = 2 * int(current_matrix / 2)
                folder_number = 0
            elif folder == 'Soft':
                captured_level = max_levels[1][current_model][current_matrix]
                marginal_distribution = marginals[1][current_model][current_matrix]
                n1_distribution = histograms[1][current_model][current_matrix]
                mean_value = means[1][current_model][current_matrix]
                row = 2 * int(current_matrix / 2) + 1
                folder_number = 1

            col = 2 * (current_matrix % 2)

            if captured_level != 0:
                captured_range = captured_level - dimension_value + 1

                if (folder == 'Soft' and current_matrix == 3 and current_model == 1) or (folder == 'Hard' and current_matrix == 3 and current_model == 0):
                    h_map = graphs[row, col].imshow(marginal_distribution, cmap="Blues", interpolation='none')
                    c_bar = fig.colorbar(h_map, ax=graphs[:, 2:], location='bottom', shrink=1)
                    c_bar.outline.set_visible(False)
                    graphs[row, col].plot(mean_value[1] - 1, mean_value[2] - 1, "d", color="black")
                    graphs[row, col].plot(original_means[current_model][0] - 1, original_means[current_model][1] - 1, "^", color="black")
                else:
                    h_map = graphs[row, col].imshow(marginal_distribution, cmap="Greens", interpolation='none', vmin=0, vmax=marginal_colour_max[current_model])
                    graphs[row, col].plot(mean_value[1] - 1, mean_value[2] - 1, "d", color="blue")
                    graphs[row, col].plot(original_means[current_model][0] - 1, original_means[current_model][1] - 1, "^", color="blue")
                if row == 0 and col == 0:
                    c_bar = fig.colorbar(h_map, ax=graphs[:, :2], location='bottom', shrink=1)
                    c_bar.outline.set_visible(False)

                graphs[row, col].set_facecolor('white')
                graphs[row, col].spines['bottom'].set_color('white')
                graphs[row, col].spines['top'].set_color('white')
                graphs[row, col].spines['right'].set_color('white')
                graphs[row, col].spines['left'].set_color('white')
                graphs[row, col].set_xlabel('$n_{2}$')
                graphs[row, col].set_ylabel('$n_{3}$')
                graphs[row, col].set_xticks(ticks)
                graphs[row, col].set_xticklabels(ticks + 1)
                graphs[row, col].set_yticks(ticks)
                graphs[row, col].set_yticklabels(ticks + 1, rotation=90)
                graphs[row, col].invert_yaxis()
                graphs[row, col].set_xlim(-0.5, 15)
                graphs[row, col].set_ylim(-0.5, 15)
                if row == 0:
                    graphs[row, col].set_title('Marginal of $n_{2}$, $n_{3}$')

                ticks = np.arange(-1, 15, 5)
                ticks[0] = 0

                graphs[row, col + 1].bar([i for i in range(captured_range)], n1_distribution, width=1, color="Green")
                graphs[row, col + 1].axvline(x=mean_value[0] - 1, color='blue')
                graphs[row, col + 1].set_xlabel('$n_{1}$')
                graphs[row, col + 1].set_xticks(ticks)
                graphs[row, col + 1].set_xticklabels(ticks + 1)
                graphs[row, col + 1].set_xlim(-1, 13)
                graphs[row, col + 1].set_ylim(0, 1)  # histogram_max[current_model] + 0.02)
                graphs[row, col + 1].set(aspect=14)
                if row == 0:
                    graphs[row, col + 1].set_title('Marginal of $n_{1}$')

    if current_model == 0:
        graphs[3, 3].axis('off')
        graphs[3, 2].axis('off')
    fig.savefig("QSD-type-{}.pdf".format(current_model + 1))

    for row in range(4):
        for col in range(4):
            graphs[row, col].clear()
    fig.clear()
    plt.close(fig='all')

captured_level = 179
fig, graphs = plt.subplots(4, 4, figsize=(16, 16), constrained_layout=True)

marginal_colour_max = np.array([])

for folder in experiments:
    for current_matrix in range(4):
        with open('{0}/Gillespie/Data-{1}.bin'.format(folder, current_matrix), 'rb') as file:
            current_data = pickle.load(file)

        marginal = np.zeros((179, 179))
        for i in range(current_data.shape[0]):
            for j in range(current_data.shape[1]):
                for k in range(current_data.shape[2]):
                    marginal[i, j] += current_data[k, j, i]
        marginal = marginal / marginal.sum()
        if folder == 'Soft' and current_matrix == 3:
            marginal_colour_max = np.append(marginal_colour_max, 0)
        else:
            marginal_colour_max = np.append(marginal_colour_max, marginal.max())

marginal_colour_max = marginal_colour_max.max()

for current_matrix in range(4):
    for folder in experiments:
        with open('{0}/Gillespie/Data-{1}.bin'.format(folder, current_matrix), 'rb') as file:
            current_data = pickle.load(file)
        if folder == 'Hard':
            row = 2 * int(current_matrix / 2)
        elif folder == 'Soft':
            row = 2 * int(current_matrix / 2) + 1

        col = 2 * (current_matrix % 2)

        mean_value = [0, 0, 0]
        marginal = np.zeros((179, 179))
        n1_marginal = np.zeros(179)
        for i in range(current_data.shape[0]):
            for j in range(current_data.shape[1]):
                for k in range(current_data.shape[2]):
                    marginal[i, j] += current_data[k, j, i]
                    n1_marginal[i] += current_data[i, j, k]
                    mean_value[0] += (i + 1) * current_data[i, j, k]
                    mean_value[1] += (j + 1) * current_data[i, j, k]
                    mean_value[2] += (k + 1) * current_data[i, j, k]

        mean_value[0] = mean_value[0] / current_data.sum()
        mean_value[1] = mean_value[1] / current_data.sum()
        mean_value[2] = mean_value[2] / current_data.sum()

        marginal_dist = marginal / marginal.sum()
        n1_marginal_dist = n1_marginal / n1_marginal.sum()

        if captured_level != 0:
            captured_range = captured_level - dimension_value + 1

            if folder == 'Soft' and current_matrix == 3:
                h_map = graphs[row, col].imshow(marginal_dist, cmap="Blues", interpolation='none')
                c_bar = fig.colorbar(h_map, ax=graphs[:, 2:], location='bottom', shrink=1)
                c_bar.outline.set_visible(False)
                graphs[row, col].plot(mean_value[1] - 1, mean_value[2] - 1, "d", color="black")
            else:
                h_map = graphs[row, col].imshow(marginal_dist, cmap="Greens", interpolation='none', vmin=0, vmax=marginal_colour_max)
                graphs[row, col].plot(mean_value[1] - 1, mean_value[2] - 1, "d", color="blue")
            if row == 0 and col == 0:
                c_bar = fig.colorbar(h_map, ax=graphs[:, :2], location='bottom', shrink=1)
                c_bar.outline.set_visible(False)

            graphs[row, col].set_facecolor('white')
            graphs[row, col].spines['bottom'].set_color('white')
            graphs[row, col].spines['top'].set_color('white')
            graphs[row, col].spines['right'].set_color('white')
            graphs[row, col].spines['left'].set_color('white')
            graphs[row, col].set_xlabel('$n_{2}$')
            graphs[row, col].set_ylabel('$n_{3}$')
            graphs[row, col].set_xticks(ticks)
            graphs[row, col].set_xticklabels(ticks + 1)
            graphs[row, col].set_yticks(ticks)
            graphs[row, col].set_yticklabels(ticks + 1, rotation=90)
            graphs[row, col].invert_yaxis()
            graphs[row, col].set_xlim(-0.5, 15)
            graphs[row, col].set_ylim(-0.5, 15)
            if row == 0:
                graphs[row, col].set_title('Marginal of $n_{2}$, $n_{3}$')

            ticks = np.arange(-1, 15, 5)
            ticks[0] = 0

            graphs[row, col + 1].bar([i for i in range(n1_marginal_dist.shape[0])], n1_marginal_dist, width=1, color="Green")
            graphs[row, col + 1].axvline(x=mean_value[0] - 1, color='blue')
            graphs[row, col + 1].set_xlabel('$n_{1}$')
            graphs[row, col + 1].set_xticks(ticks)
            graphs[row, col + 1].set_xticklabels(ticks + 1)
            graphs[row, col + 1].set_xlim(-1, 13)
            graphs[row, col + 1].set_ylim(0, 1)  # histogram_max[current_model] + 0.02)
            graphs[row, col + 1].set(aspect=14)
            if row == 0:
                graphs[row, col + 1].set_title('Marginal of $n_{1}$')

fig.savefig("QSD-G.pdf")

for row in range(4):
    for col in range(4):
        graphs[row, col].clear()
fig.clear()
plt.close(fig='all')

with open('Truncated_levels.bin', 'wb') as file:
    pickle.dump(max_levels, file)
