# %% Packages


import pickle
from copy import deepcopy
from distutils.spawn import find_executable

import matplotlib.pyplot as plt

from homeostatic import absorption_distribution_value

if find_executable("latex"):
    plt.rcParams.update({"text.usetex": True})
plt.rcParams["font.family"] = "serif"
plt.rcParams["mathtext.fontset"] = "dejavuserif"

# %% Loading data


initial_state = [8, 8]
experiments = ["Hard", "Soft"]
matrices = 4

hard_data = []
soft_data = []
hard_indices = []
soft_indices = []

for current_matrix in range(matrices):

    absorption_values_hard = [[], [], []]
    absorption_values_soft = [[], [], []]

    for folder in experiments:
        with open(f"{folder}/Parameters-{current_matrix}.bin", "rb") as file:
            load_data = pickle.load(file)

            dimension_value = load_data[1]
            max_level_value = load_data[2]
            stimulus_value = load_data[5]

            del load_data

        with open(f"{folder}/Data-{current_matrix}.bin", "rb") as file:
            distribution = pickle.load(file)

        absorption_index = [
            i + 1 for i in range(max_level_value - initial_state[0] - initial_state[1])
        ]

        for i in range(max_level_value - initial_state[0] - initial_state[1]):
            plotted_state = [i + 1, initial_state[0], initial_state[1]]

            if folder == "Hard":
                for current_clone in range(3):
                    absorption_values_hard[current_clone].append(
                        absorption_distribution_value(
                            current_clone,
                            plotted_state,
                            dimension_value,
                            max_level_value,
                            distribution,
                        )
                    )
            else:
                for current_clone in range(3):
                    absorption_values_soft[current_clone].append(
                        absorption_distribution_value(
                            current_clone,
                            plotted_state,
                            dimension_value,
                            max_level_value,
                            distribution,
                        )
                    )

        if folder == "Hard":
            hard_data.append(deepcopy(absorption_values_hard))
            hard_indices.append(deepcopy(absorption_index))
        else:
            soft_data.append(deepcopy(absorption_values_soft))
            soft_indices.append(deepcopy(absorption_index))

# %% Generating figure


h = 3.5
w = 3 * h
lw = 1.0
displacement = 0.25

fig, graphs = plt.subplots(1, 3, figsize=(w, (1 / (1 - displacement)) * h))
fig.tight_layout(rect=(0.025, displacement, 1, 0.975))

label_size = 18
title_size = 20

bpattern = [6, 4]
cpattern = [7, 5, 2, 5]
dpattern = [2, 4]

# Empty plots for general legend (white plot included to align labels)
graphs[0].plot([], [], "-", color="k", label="$\\textrm{Scenario } (\\textrm{a})$")
graphs[0].plot([], [], "-", color="w", label="$\\textrm{ }$")
graphs[0].plot(
    [], [], dashes=bpattern, color="k", label="$\\textrm{Scenario } (\\textrm{b})$"
)
graphs[0].plot([], [], "s", color="g", label="$\\textrm{Hard niche}$")
graphs[0].plot(
    [], [], dashes=cpattern, color="k", label="$\\textrm{Scenario } (\\textrm{c})$"
)
graphs[0].plot([], [], "s", color="b", label="$\\textrm{Soft niche}$")
graphs[0].plot(
    [], [], dashes=dpattern, color="k", label="$\\textrm{Scenario } (\\textrm{d})$"
)

graphs[0].plot(soft_indices[0], soft_data[0][0], "-", lw=lw, color="b")
graphs[0].plot(soft_indices[1], soft_data[1][0], dashes=bpattern, lw=lw, color="b")
graphs[0].plot(soft_indices[2], soft_data[2][0], dashes=cpattern, lw=lw, color="b")
graphs[0].plot(soft_indices[3], soft_data[3][0], dashes=dpattern, lw=lw, color="b")
graphs[0].plot(hard_indices[0], hard_data[0][0], "-", lw=lw, color="g")
graphs[0].plot(hard_indices[1], hard_data[1][0], dashes=bpattern, lw=lw, color="g")
graphs[0].plot(hard_indices[2], hard_data[2][0], dashes=cpattern, lw=lw, color="g")
graphs[0].plot(hard_indices[3], hard_data[3][0], dashes=dpattern, lw=lw, color="g")
graphs[0].set_title("$\\mathcal{U}^{1}(n_{1})$", fontsize=title_size)
graphs[0].set_xlabel("$n_{1}$", fontsize=label_size)
graphs[0].set_ylabel("$\\textrm{Probability}$", fontsize=label_size)
graphs[0].set_facecolor("white")
graphs[0].set_ylim(0, 1)
graphs[0].set_xlim(1, max_level_value - initial_state[0] - initial_state[1])
graphs[0].tick_params(axis="both", labelsize=11)

graphs[1].plot(soft_indices[0], soft_data[0][1], "-", lw=lw, color="b")
graphs[1].plot(soft_indices[1], soft_data[1][1], dashes=bpattern, lw=lw, color="b")
graphs[1].plot(soft_indices[2], soft_data[2][1], dashes=cpattern, lw=lw, color="b")
graphs[1].plot(soft_indices[3], soft_data[3][1], dashes=dpattern, lw=lw, color="b")
graphs[1].plot(hard_indices[0], hard_data[0][1], "-", lw=lw, color="g")
graphs[1].plot(hard_indices[1], hard_data[1][1], dashes=bpattern, lw=lw, color="g")
graphs[1].plot(hard_indices[2], hard_data[2][1], dashes=cpattern, lw=lw, color="g")
graphs[1].plot(hard_indices[3], hard_data[3][1], dashes=dpattern, lw=lw, color="g")
graphs[1].set_title("$\\mathcal{U}^{2}(n_{1})$", fontsize=title_size)
graphs[1].set_xlabel("$n_{1}$", fontsize=label_size)
graphs[1].set_facecolor("white")
graphs[1].set_ylim(0, 1)
graphs[1].set_xlim(1, max_level_value - initial_state[0] - initial_state[1])
graphs[1].tick_params(axis="both", labelsize=11, labelleft=False)

graphs[2].plot(soft_indices[0], soft_data[0][2], "-", lw=lw, color="b")
graphs[2].plot(soft_indices[1], soft_data[1][2], dashes=bpattern, lw=lw, color="b")
graphs[2].plot(soft_indices[2], soft_data[2][2], dashes=cpattern, lw=lw, color="b")
graphs[2].plot(soft_indices[3], soft_data[3][2], dashes=dpattern, lw=lw, color="b")
graphs[2].plot(hard_indices[0], hard_data[0][2], "-", lw=lw, color="g")
graphs[2].plot(hard_indices[1], hard_data[1][2], dashes=bpattern, lw=lw, color="g")
graphs[2].plot(hard_indices[2], hard_data[2][2], dashes=cpattern, lw=lw, color="g")
graphs[2].plot(hard_indices[3], hard_data[3][2], dashes=dpattern, lw=lw, color="g")
graphs[2].set_title("$\\mathcal{U}^{3}(n_{1})$", fontsize=title_size)
graphs[2].set_xlabel("$n_{1}$", fontsize=label_size)
graphs[2].set_facecolor("white")
graphs[2].set_ylim(0, 1)
graphs[2].set_xlim(1, max_level_value - initial_state[0] - initial_state[1])
graphs[2].tick_params(axis="both", labelsize=11, labelleft=False)

handles, labels = graphs[0].get_legend_handles_labels()
fig.legend(
    handles,
    labels,
    ncol=4,
    loc="upper center",
    bbox_to_anchor=(0.5, displacement),
    fontsize=label_size,
    borderaxespad=1,
)

fig.savefig("First-extinction.pdf")

for col in range(3):
    graphs[col].clear()
fig.clear()
plt.close(fig="all")
