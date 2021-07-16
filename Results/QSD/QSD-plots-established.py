#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# %% Packages


from scipy.special import comb
from copy import deepcopy
import numpy as np
import pickle
import tikzplotlib
import seaborn as sns
import matplotlib.pyplot as plt

sns.set(font='serif')
sns.set_style("ticks")
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


# %% Plotting distributions


epsilon = 0.0001
means = []

with open("Established/Parameters.bin", 'rb') as file:
    load_data = pickle.load(file)

    dimension_value = load_data[1]
    max_level_value = load_data[2]
    stimulus_value = load_data[5]

    del load_data

all_marginals = []
all_means = []

for model in range(2):
    with open("Established/Data-{}-0.bin".format(model + 1), 'rb') as file:
        data = pickle.load(file)

    captured = 0
    for captured_level in range(len(data)):
        captured += sum(data[captured_level])
        if captured >= 1 - epsilon:
            captured_level += 1
            break

    captured_range = captured_level - dimension_value + 1

    mean_value = [0, 0]
    distribution = np.zeros((captured_range, captured_range))
    for k in range(distribution.shape[0]):
        for m in range(distribution.shape[1]):
            if k + m <= captured_level:
                distribution[k][m] += data[k + m][level_position(k + m + dimension_value, dimension_value, [k + 1, m + 1])]
                mean_value[0] += (k + 1) * distribution[k][m]
                mean_value[1] += (m + 1) * distribution[k][m]

    marginals = [np.zeros(captured_range), np.zeros(captured_range)]
    for k in range(distribution.shape[0]):
        for m in range(distribution.shape[1]):
            marginals[0][k] += distribution[k, m]
            marginals[1][m] += distribution[k, m]

    means.append(mean_value)

    all_marginals.append(deepcopy(marginals))
    all_means.append(deepcopy(means))

    fig, graph = plt.subplots(1, 1, tight_layout=True)

    # ticks = np.arange(-1, captured_range, 5)
    ticks = np.arange(-1, 18, 5)
    ticks[0] = 0

    # # Seaborn
    # h_map = sns.heatmap(distribution, cmap="Greens")
    # h_map.set_xlabel('$n_{2}$')
    # h_map.set_ylabel('$n_{3}$')
    # h_map.set_xticks(ticks + 0.5)
    # h_map.set_xticklabels(ticks + 1)
    # h_map.set_yticks(ticks + 0.5)
    # h_map.set_yticklabels(ticks + 1)  # , rotation=0)
    # h_map.scatter(mean_value[0] - 1, mean_value[1] - 1, marker='s', s=50, color='yellow')
    # h_map.invert_yaxis()

    # # Plot
    h_map = graph.imshow(distribution, cmap="Greens", interpolation='none')
    c_bar = fig.colorbar(h_map)
    c_bar.outline.set_visible(False)
    graph.plot(mean_value[0] - 1, mean_value[1] - 1, "^", color="yellow")
    graph.set_facecolor('white')
    graph.spines['bottom'].set_color('white')
    graph.spines['top'].set_color('white')
    graph.spines['right'].set_color('white')
    graph.spines['left'].set_color('white')
    graph.set_xlabel('$n_{2}$')
    graph.set_ylabel('$n_{3}$')
    graph.set_xlim(-0.5, 17.5)
    graph.set_ylim(-0.5, 17.5)
    graph.set_xticks(ticks)
    graph.set_xticklabels(ticks + 1)
    graph.set_yticks(ticks)
    graph.set_yticklabels(ticks + 1, rotation=90)
    # graph.invert_yaxis()
    plt.title('QSD approximation using $\\mathcal{X}^{' + '({})'.format(model + 1) + '}$')

    fig.savefig("Established/QSD-{}.pdf".format(model + 1))
    # tikzplotlib.clean_figure()
    # tikzplotlib.save("Established/QSD-{}.tex".format(model + 1))

    graph.clear()
    fig.clear()
    plt.close(fig='all')


with open("Established/Data.bin", 'rb') as file:
    data = pickle.load(file)

marginal_g1 = np.zeros(179)
marginal_g2 = np.zeros(179)
for i in range(data.shape[0]):
    for j in range(data.shape[1]):
            marginal_g1[i] += data[i, j]
            marginal_g2[j] += data[i, j]

marginal_g1 = marginal_g1 / marginal_g1.sum()
marginal_g2 = marginal_g2 / marginal_g2.sum()

fig, graph = plt.subplots(1, 1, constrained_layout=True, figsize=(4.5, 3.5))

width = 0.25
ticks = np.arange(0, 18)
ticks_1 = np.arange(0, all_marginals[0][0].shape[0])
ticks_2 = np.arange(0, all_marginals[1][0].shape[0])
ticks_g = np.arange(0, marginal_g1.shape[0])
# bars_1 = graph.bar(ticks_1 - width, all_marginals[0][0], width, label='$\mathcal{X}^{(1)}$', color='Green')
# gill = graph.bar(ticks_g, marginal_g1, width, label='Gillespie', color='red')
# bars_2 = graph.bar(ticks_2 + width, all_marginals[1][0], width, label='$\mathcal{X}^{(2)}$', color='blue')
bars_1 = graph.bar(ticks_1 - (3/2) * width, all_marginals[0][0], width, label='$\mathcal{X}^{(1)}$', color='Green', linewidth=0.8)
gill = graph.bar(ticks_g - width/2, marginal_g1, width, label='$\\textrm{Gillespie}$', color='red', linewidth=0.8)
bars_2s = graph.bar(ticks_2 - 1 + width/2, all_marginals[1][0], width, label='$\\textrm{Shifted } \mathcal{X}^{(2)}$', color='blue', hatch='///', linewidth=0.8)
bars_2 = graph.bar(ticks_2 + (3/2) * width, all_marginals[1][0], width, label='$\mathcal{X}^{(2)}$', color='blue', linewidth=0.8)
graph.set_xticks(ticks)
graph.set_xticklabels(ticks + 1)
graph.set_xlim(-1, 18)
graph.set_ylim(0, 0.2)
graph.legend(loc='upper right')  # , fontsize='large')
graph.set_xlabel('$n_{2}$')
graph.set_title('$\\textrm{Marginal distribution of } n_{2}$')

fig.savefig("Established/QSD-n2.pdf")
fig.clear()
plt.close(fig='all')

fig, graph = plt.subplots(1, 1, constrained_layout=True, figsize=(4.5, 3.5))

# width = 1/3
# ticks = np.arange(0, 18)
# ticks_1 = np.arange(0, all_marginals[0][1].shape[0])
# ticks_2 = np.arange(0, all_marginals[1][1].shape[0])
# bars_1 = graph.bar(ticks_1 - width, all_marginals[0][1], width, label='$\mathcal{X}^{(1)}$', color='Green')
# bars_2 = graph.bar(ticks_2, all_marginals[1][1], width, label='$\mathcal{X}^{(2)}$', color='blue')
# bars_3 = graph.bar(ticks_2 + width, all_marginals[1][1], width, label='$\mathcal{X}^{(2)}$', color='blue')

width = 0.25
ticks = np.arange(0, 18)
ticks_1 = np.arange(0, all_marginals[0][1].shape[0])
ticks_2 = np.arange(0, all_marginals[1][1].shape[0])
ticks_g = np.arange(0, marginal_g2.shape[0])
bars_1 = graph.bar(ticks_1 - (3/2) * width, all_marginals[0][1], width, label='$\mathcal{X}^{(1)}$', color='Green', linewidth=0.8)
gill = graph.bar(ticks_g - width/2, marginal_g2, width, label='$\\textrm{Gillespie}$', color='red', linewidth=0.8)
bars_2s = graph.bar(ticks_2 - 1 + width/2, all_marginals[1][1], width, label='$\\textrm{Shifted } \mathcal{X}^{(2)}$', color='blue', hatch='///', linewidth=0.8)
bars_2 = graph.bar(ticks_2 + (3/2) * width, all_marginals[1][1], width, label='$\mathcal{X}^{(2)}$', color='blue', linewidth=0.8)
graph.set_xticks(ticks)
graph.set_xticklabels(ticks + 1)
graph.set_xlim(-1, 18)
graph.set_ylim(0, 0.2)
graph.legend(loc='upper right')
graph.set_xlabel('$n_{3}$')
graph.set_title('$\\textrm{Marginal distribution of } n_{3}$')

fig.savefig("Established/QSD-n3.pdf")
fig.clear()
plt.close(fig='all')


with open("Established/Means.bin", 'wb') as file:
    pickle.dump(means, file)
