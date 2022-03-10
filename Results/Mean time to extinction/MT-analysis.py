#%% Packages

import os
import sys

sys.path.append(
    os.path.join(os.path.join(os.path.dirname(__file__), os.pardir), os.pardir)
)

import csv
import pickle

import numpy as np

from homeostatic.definitions import level_position, level_states

#%% Data analysis


experiments = ["Hard", "Soft"]
scenarios = ["a", "b", "c", "d"]
starting_state = [4, 8, 8]

state_position = level_position(
    sum(starting_state), len(starting_state), starting_state
)
for i in range(len(starting_state), sum(starting_state)):
    state_position += len(level_states(i, 3))

with open("Results.csv", "a", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["Established clonotypes"])
    writer.writerow(["Minimum", "Maximum", "Mean", "SD"])

    with open("Established/Data.bin", "rb") as data_file:
        data = pickle.load(data_file)
        row_data = [
            round(data.min(), 5),
            round(data.max(), 5),
            round(np.mean(data), 5),
            round(np.std(data), 5),
        ]
        writer.writerow(row_data)

    writer.writerow([""])

    writer.writerow(["3 Clonotypes"])
    writer.writerow(
        [
            "Niche",
            "Scenario",
            "Minimum",
            "Maximum",
            "Mean",
            "SD",
            f"n0 = ({starting_state[0]}, {starting_state[1]}, {starting_state[2]})",
        ]
    )

    for current_experiment in experiments:
        for current_matrix in range(4):
            with open(
                f"{current_experiment}/Matrix-{current_matrix}/Data.bin", "rb"
            ) as data_file:
                data = pickle.load(data_file)
                row_data = [
                    f"{current_experiment} niche",
                    scenarios[current_matrix],
                    round(data.min(), 5),
                    round(data.max(), 5),
                    round(np.mean(data), 5),
                    round(np.std(data), 5),
                    round(data[state_position], 5),
                ]
                writer.writerow(row_data)
