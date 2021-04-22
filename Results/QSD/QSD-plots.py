#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# %% Packages


from scipy.special import comb
import numpy as np
import pickle
import tikzplotlib
import seaborn as sns
import matplotlib.pyplot as plt

sns.set(font='serif')
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


# experiments = ['Established']  # , 'Extinction-balanced', 'Extinction-competition']
experiments = ['Hard', 'Soft']
epsilon = 0.0001

for folder in experiments:
    max_levels = [[] for _ in range(2)]
    max_values = [[] for _ in range(2)]
    for current_model in range(2):
        matrices = 4  # 3
        # if folder == 'Extinction-competition':
        #     matrices = 4
        for current_matrix in range(matrices):
            # open("{0}/Model-{1}/QSD-{2}/Parameters.bin".format(folder, i, j), 'rb')
            # file = open("{0}{1}/QSD-0-{2}/Parameters.bin".format(folder, current_model, current_matrix + 1), 'rb')
            try:
                file = open("{0}/Model-{1}/QSD-{2}-{3}/Parameters.bin".format(folder, current_model + 1, folder[0], current_matrix), 'rb')
                load_data = pickle.load(file)
                file.close()

                dimension_value = load_data[1]
                max_level_value = load_data[2]
                stimulus_value = load_data[5]

                del load_data

                # open("{0}/Model-{1}/QSD-{2}/Data-{3}-{4}.bin".format(folder, i, j, i, j), 'rb')
                # file = open("{0}{1}/QSD-0-{2}/Data-{3}-{4}.bin".format(folder, current_model, current_matrix + 1, current_model, current_matrix + 1), 'rb')
                file = open("{0}/Model-{1}/QSD-{2}-{3}/Data-{4}-{5}.bin".format(folder, current_model + 1, folder[0], current_matrix, current_model + 1, current_matrix), 'rb')
                data = pickle.load(file)
                file.close()

                level_lists = []

                captured = 0
                for captured_level in range(len(data)):
                    captured += sum(data[captured_level])
                    if captured >= 1 - epsilon:
                        captured_level += 1
                        break

                captured_range = captured_level - dimension_value + 1

                distribution = np.zeros((captured_range, captured_range, captured_range))
                for k in range(len(distribution)):
                    for m in range(len(distribution[k])):
                        for n in range(len(distribution[k][m])):
                            if k + m + n <= captured_level:
                                distribution[k][m][n] += data[k + m + n][level_position(k + m + n + dimension_value, dimension_value, [k + 1, m + 1, n + 1])]

                max_levels[current_model].append(captured_level)

                max_values[current_model].append([[np.where(distribution == np.amax(distribution))[0][0], np.where(distribution == np.amax(distribution))[1][0], np.where(distribution == np.amax(distribution))[2][0]], distribution.max()])

                indexes = [i + 1 for i in range(captured_range)]
                X, Y = np.meshgrid(indexes, indexes)

                marginal_distributions = []

                marginal_1v2 = np.zeros((captured_range, captured_range))
                for d in range(captured_range):
                    for m in range(captured_range):
                        for n in range(captured_range):
                            marginal_1v2[m][d] += distribution[d][m][n]

                marginal_distributions.append(marginal_1v2)

                marginal_1v3 = np.zeros((captured_range, captured_range))
                for d in range(captured_range):
                    for m in range(captured_range):
                        for n in range(captured_range):
                            marginal_1v3[m][d] += distribution[d][n][m]

                marginal_distributions.append(marginal_1v3)

                marginal_2v3 = np.zeros((captured_range, captured_range))
                for d in range(captured_range):
                    for m in range(captured_range):
                        for n in range(captured_range):
                            marginal_2v3[m][d] += distribution[n][d][m]

                marginal_distributions.append(marginal_2v3)

                for current_marginal in range(len(marginal_distributions)):
                    raw_levels = np.linspace(marginal_distributions[current_marginal].min(), marginal_distributions[current_marginal].max(), 5000)
                    level_values = [0.15, 0.35, 0.55, 0.75, 0.95]
                    refined_levels = []
                    for current_raw_level in range(raw_levels.shape[0]):
                        total = 0.0
                        for row in range(len(marginal_distributions[current_marginal])):
                            for col in range(len(marginal_distributions[current_marginal][row])):
                                if marginal_distributions[current_marginal][row][col] <= raw_levels[current_raw_level]:
                                    total += marginal_distributions[current_marginal][row][col]
                        if len(refined_levels) != len(level_values):
                            if level_values[len(refined_levels)] <= total:  # 0.1 * (len(refined_levels) + 1) <= total and 0.1 * (len(refined_levels) + 1) < 1.0:
                                refined_levels.append(raw_levels[current_raw_level])

                    level_lists.append(refined_levels)

                fig, graph = plt.subplots(1, 1)

                CS = graph.contour(X, Y, marginal_distributions[0], level_lists[0], colors='black', linestyles='solid', alpha=1)
                CS1 = graph.contour(X, Y, marginal_distributions[1], level_lists[1], colors='red', linestyles='dashed', alpha=1)
                CS2 = graph.contour(X, Y, marginal_distributions[2], level_lists[2], colors='blue', linestyles='dotted', alpha=1)

                h1, _ = CS.legend_elements()
                h2, _ = CS1.legend_elements()
                h3, _ = CS2.legend_elements()

                graph.legend([h1[0], h2[0], h3[0]], ['$\mathbb{P}_{1,2}$', '$\mathbb{P}_{1,3}$', '$\mathbb{P}_{2,3}$'], facecolor='white', framealpha=1, fontsize=13)

                # CS = graph.contourf(X, Y, marginal_distributions[2], level_lists[2], cmap=plt.get_cmap('Greens'))
                # h1, _ = CS.legend_elements()
                # fmt = {}
                # level_labels = ['{}'.format(value) for value in level_values]
                # for lev, s in zip(CS.levels, level_labels):
                #     fmt[lev] = s
                # graph.clabel(CS, fmt=fmt, fontsize=10)
                # graph.legend([h1[0]], ['$\mathbb{P}_{2,3}$'], facecolor='white', framealpha=1, fontsize=13)

                graph.set_facecolor('white')
                graph.spines['bottom'].set_color('gray')
                graph.spines['top'].set_color('gray')
                graph.spines['right'].set_color('gray')
                graph.spines['left'].set_color('gray')
                graph.set_xlim(1, max_levels[current_model][-1])
                graph.set_ylim(1, max_levels[current_model][-1])
                # graph.yaxis.grid(color='gray')
                # graph.xaxis.grid(color='gray')
                plt.title('Marginal QSD for $\\mathcal{X}^{' + '({})'.format(current_model + 1) + '}$')

                # fig.savefig('{0}/QSD-{1}-{2}.pdf'.format(folder, i, j))
                # tikzplotlib.save("{0}/QSD-{1}-{2}.tex".format(folder, i, j))
                # fig.savefig("{0}{1}/QSD-{2}-{3}.pdf".format(folder, current_model, current_model, current_matrix + 1))

                fig.savefig("{0}/Model-{1}/QSD-{2}-{3}.pdf".format(folder, current_model + 1, folder[0], current_matrix))
                tikzplotlib.save("{0}/Model-{1}/QSD-{2}-{3}.tex".format(folder, current_model + 1, folder[0], current_matrix))

                graph.clear()
                fig.clear()
                plt.close(fig='all')

            except FileNotFoundError:
                max_levels[current_model].append(0)
                max_values[current_model].append(0)

    pickle.dump(max_levels, open('{}/Truncated_levels.bin'.format(folder), 'wb'))
    pickle.dump(max_values, open('{}/Maximum_values.bin'.format(folder), 'wb'))
