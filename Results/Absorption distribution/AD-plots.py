# %% Packages

import os
import sys

sys.path.append(
    os.path.join(os.path.join(os.path.dirname(__file__), os.pardir), os.pardir)
)

import pickle
from copy import deepcopy
from distutils.spawn import find_executable

import matplotlib.pyplot as plt
import numpy as np

from homeostatic.definitions import absorption_distribution

if find_executable("latex"):
    plt.rcParams.update({"text.usetex": True})
    plt.rcParams["text.latex.preamble"] = r"\usepackage{graphicx}\usepackage{eucal}"
plt.rcParams["font.family"] = "serif"
plt.rcParams["mathtext.fontset"] = "dejavuserif"
plt.rcParams["figure.constrained_layout.use"] = True

# %% Loading data


plotted_state = [4, 8, 8]
experiments = ["Hard", "Soft"]
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
        with open(f"{folder}/Parameters-{current_matrix}.bin", "rb") as file:
            load_data = pickle.load(file)

        dimension_value = load_data[1]
        max_level_value = load_data[2]
        stimulus_value = load_data[5]

        del load_data

        with open(f"{folder}/Data-{current_matrix}.bin", "rb") as file:
            distribution = pickle.load(file)

        absorption_distributions = [
            absorption_distribution(
                i, plotted_state, dimension_value, max_level_value, distribution
            )
            for i in range(3)
        ]
        absorption_values = [d.sum() for d in absorption_distributions]
        matrix_means = []

        # Conditional probabilities
        for i, d in enumerate(absorption_distributions):
            absorption_distributions[i] = d / d.sum()

        max_value = max(
            absorption_distributions[0].max(),
            absorption_distributions[1].max(),
            absorption_distributions[2].max(),
        )

        if folder == "Hard":
            max_values[0] = np.append(max_values[0], max_value)
        elif folder == "Soft":
            max_values[1] = np.append(max_values[1], max_value)

        for clone in range(3):
            mean_value = [0, 0]
            for row in range(absorption_distributions[clone].shape[0]):
                for col in range(absorption_distributions[clone].shape[1]):
                    mean_value[0] += (row + 1) * absorption_distributions[clone][
                        row, col
                    ]
                    mean_value[1] += (col + 1) * absorption_distributions[clone][
                        row, col
                    ]
            matrix_means.append(deepcopy(mean_value))

        folder_distributions.append(deepcopy(absorption_distributions))
        folder_means.append(deepcopy(matrix_means))
        folder_pie_values.append(deepcopy(absorption_values))

    distributions.append(deepcopy(folder_distributions))
    means.append(deepcopy(folder_means))
    pie_values.append(deepcopy(folder_pie_values))

max_values = [max_values[0].max(), max_values[1].max()]

# %% Generating figure


wspacing = 0.05
hspacing = 1.5 * wspacing
scenario_names = [
    "$\\textbf{(a)}$",
    "$\\textbf{(b)}$",
    "$\\textbf{(c)}$",
    "$\\textbf{(d)}$",
]
height = 32
width = 0.55 * height

max_axis = 14.5
ticks = np.arange(-1, int(max_axis + 0.5), 5)
ticks[0] = 0

tick_size = 33
label_size = 39
title_size = 43
mark_size = 10
side_label_ratio = 5

fig = plt.figure(figsize=(width, height))
top_bottom_figs = fig.subfigures(2, 1, height_ratios=[24, 1.15], hspace=0)
sub_figs = top_bottom_figs[0].subfigures(
    4, 2, width_ratios=[side_label_ratio, 100 - side_label_ratio], hspace=0, wspace=0
)
bar_figs = top_bottom_figs[1].subfigures(1, 2, hspace=0, wspace=0)
subfig_list = np.empty(4, dtype=object)
subfig_h_maps = np.empty(2, dtype=object)

for current_matrix in range(matrices):
    sub_figs[current_matrix, 1].suptitle(
        "{\\_}",
        fontsize=1,
        horizontalalignment="left",
        visible=False,
    )
    sub_figs[current_matrix, 0].text(
        0.5,
        0.5,
        scenario_names[current_matrix],
        fontsize=title_size,
        ha="center",
        va="center",
    )
    subfig_list[current_matrix] = sub_figs[current_matrix, 1].subplots(2, 4)

    for folder in experiments:
        subfig_row = 0
        subfig_col = 0
        folder_number = 0
        if folder == "Hard":
            colour_max = max_values[0]
            distribution = distributions[0][current_matrix]
            colour_map = "Greens"
        else:
            subfig_row += 1
            folder_number += 1
            colour_max = max_values[1]
            distribution = distributions[1][current_matrix]
            colour_map = "Blues"

        for clone in range(3):
            mean_value = means[folder_number][current_matrix][clone]
            col = subfig_col + clone
            labels = [1, 2, 3]
            labels.pop(labels.index(clone + 1))

            if current_matrix == folder_number == clone == 0:
                subfig_h_maps[0] = subfig_list[current_matrix][subfig_row, col].imshow(
                    distribution[clone],
                    cmap=colour_map,
                    interpolation="none",
                    vmin=0,
                    vmax=colour_max,
                )
            elif current_matrix == clone == 0 and folder_number == 1:
                subfig_h_maps[1] = subfig_list[current_matrix][subfig_row, col].imshow(
                    distribution[clone],
                    cmap=colour_map,
                    interpolation="none",
                    vmin=0,
                    vmax=colour_max,
                )
            else:
                subfig_list[current_matrix][subfig_row, col].imshow(
                    distribution[clone],
                    cmap=colour_map,
                    interpolation="none",
                    vmin=0,
                    vmax=colour_max,
                )

            if folder_number == 0:
                subfig_list[current_matrix][subfig_row, col].plot(
                    plotted_state[labels[1] - 1] - 1,
                    plotted_state[labels[0] - 1] - 1,
                    "^",
                    color="#133317",
                    ms=mark_size,
                )
                subfig_list[current_matrix][subfig_row, col].plot(
                    mean_value[1] - 1,
                    mean_value[0] - 1,
                    "d",
                    color="#133317",
                    ms=mark_size,
                )
            elif folder_number == 1:
                subfig_list[current_matrix][subfig_row, col].plot(
                    plotted_state[labels[1] - 1] - 1,
                    plotted_state[labels[0] - 1] - 1,
                    "^",
                    color="#664C12",
                    ms=mark_size,
                )
                subfig_list[current_matrix][subfig_row, col].plot(
                    mean_value[1] - 1,
                    mean_value[0] - 1,
                    "d",
                    color="#664C12",
                    ms=mark_size,
                )

            subfig_list[current_matrix][subfig_row, col].set_facecolor("white")
            subfig_list[current_matrix][subfig_row, col].spines["bottom"].set_visible(
                False
            )
            subfig_list[current_matrix][subfig_row, col].spines["top"].set_visible(
                False
            )
            subfig_list[current_matrix][subfig_row, col].spines["right"].set_visible(
                False
            )
            subfig_list[current_matrix][subfig_row, col].spines["left"].set_visible(
                False
            )
            subfig_list[current_matrix][subfig_row, col].set_xlim(-0.5, max_axis)
            subfig_list[current_matrix][subfig_row, col].set_ylim(-0.5, max_axis)
            subfig_list[current_matrix][subfig_row, col].set_xticks(ticks)

            subfig_list[current_matrix][subfig_row, col].set_ylabel(
                f"$n_{labels[0]}$", fontsize=label_size
            )
            subfig_list[current_matrix][subfig_row, col].set_yticks(ticks)

            if current_matrix == 3 and subfig_row == 1:
                subfig_list[current_matrix][subfig_row, col].set_xticklabels(
                    ticks + 1, fontsize=tick_size
                )
                subfig_list[current_matrix][subfig_row, col].set_xlabel(
                    f"$n_{labels[1]}$", fontsize=label_size
                )
            elif subfig_row == 1:
                subfig_list[current_matrix][subfig_row, col].set_xticklabels(
                    ticks + 1, fontsize=tick_size, color="w"
                )
                subfig_list[current_matrix][subfig_row, col].set_xlabel(
                    f"$n_{labels[1]}$", fontsize=label_size, color="w"
                )
            else:
                subfig_list[current_matrix][subfig_row, col].set_xticklabels([])

            if col == 0:
                subfig_list[current_matrix][subfig_row, col].set_yticklabels(
                    ticks + 1, fontsize=tick_size, rotation=90
                )
            else:
                subfig_list[current_matrix][subfig_row, col].set_yticklabels([])

            if current_matrix == subfig_row == 0:
                subfig_list[current_matrix][subfig_row, col].set_title(
                    "$\\mathbf{U}^" + f"{clone + 1}$", fontsize=title_size
                )
            elif subfig_row == 0:
                subfig_list[current_matrix][subfig_row, col].set_title(
                    "$\\mathbf{U}^" + f"{clone + 1}$", fontsize=title_size, color="w"
                )

        subfig_list[current_matrix][subfig_row, 3].barh(
            np.arange(3),
            pie_values[folder_number][current_matrix],
            tick_label=[
                "$\\mathcal{U}^{1}_{\\mathbf{n}_{0}}$",
                "$\\mathcal{U}^{2}_{\\mathbf{n}_{0}}$",
                "$\\mathcal{U}^{3}_{\\mathbf{n}_{0}}$",
            ],
            color=["#E54517", "#78FF78", "#A591FF"],
        )
        subfig_list[current_matrix][subfig_row, 3].tick_params(
            axis="y", labelsize=tick_size
        )
        subfig_list[current_matrix][subfig_row, 3].tick_params(
            axis="x", labelsize=tick_size
        )
        subfig_list[current_matrix][subfig_row, 3].invert_yaxis()
        subfig_list[current_matrix][subfig_row, 3].spines["right"].set_visible(False)
        subfig_list[current_matrix][subfig_row, 3].spines["top"].set_visible(False)
        subfig_list[current_matrix][subfig_row, 3].set_xlim(0, 1)

        if subfig_row == 1:
            for current_col in range(4):
                subfig_list[current_matrix][subfig_row, current_col].tick_params(
                    labelbottom=True
                )
        else:
            for current_col in range(4):
                subfig_list[current_matrix][subfig_row, current_col].tick_params(
                    labelbottom=False
                )


left_axis = bar_figs[0].subplots(1)
left_axis.axis("off")
c_bar_l = fig.colorbar(
    subfig_h_maps[0], ax=left_axis, orientation="horizontal", fraction=0.95, aspect=40
)
c_bar_l.ax.tick_params(labelsize=tick_size)
c_bar_l.outline.set_visible(False)
c_bar_l.set_label("$\\textrm{Probability (hard niche case)}$", fontsize=label_size)

right_axis = bar_figs[1].subplots(1)
right_axis.axis("off")
c_bar_r = fig.colorbar(
    subfig_h_maps[1], ax=right_axis, orientation="horizontal", fraction=0.95, aspect=40
)
c_bar_r.ax.tick_params(labelsize=tick_size)
c_bar_r.outline.set_visible(False)
c_bar_r.set_label("$\\textrm{Probability (soft niche case)}$", fontsize=label_size)

fig.savefig("AD.pdf")
plt.close(fig="all")
