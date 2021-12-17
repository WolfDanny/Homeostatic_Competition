#%% Packages


from homeostatic import *
import math
import pickle
import os
import numpy as np
from random import uniform

#%% Parameters


new_clone_is_soft = False
max_level_value = 179
mu_value = 1.0
n_mean_value = 10
gamma_value = 1.0
realisations = 10000
time_max = 40
clones = 3
sample_value = 0  # Not used if 'clones' is 2
base_stimulus = 10
initial_cells = 10


#%% Reading Samples


if clones == 2:
    stimulus_value = [base_stimulus * gamma_value, base_stimulus * gamma_value]
    distribution = np.zeros((max_level_value, max_level_value))
    initial_state = [initial_cells, initial_cells]

    probability_values = np.genfromtxt("Samples/Established-Matrix/Matrix-2C.csv", delimiter=",")
    nu_value = np.genfromtxt("Samples/Established-Nu-Matrix/Nu-Matrix-2C.csv", delimiter=",")

if clones == 3:
    stimulus_value = [base_stimulus * gamma_value, base_stimulus * gamma_value, base_stimulus * gamma_value]
    distribution = np.zeros((max_level_value, max_level_value, max_level_value))
    initial_state = [initial_cells, initial_cells, initial_cells]

    probability_values = np.genfromtxt(f"Samples/Matrices/Matrix-{sample_value}.csv", delimiter=",")
    if sample_value < 3:
        if new_clone_is_soft:
            nu_value = np.genfromtxt("Samples/Nu-Matrices/Nu-Matrix-Soft.csv", delimiter=",")
        else:
            nu_value = np.genfromtxt("Samples/Nu-Matrices/Nu-Matrix-Hard.csv", delimiter=",")
    else:
        if new_clone_is_soft:
            nu_value = np.genfromtxt("Samples/Nu-Matrices/Nu-Matrix-Soft-(D).csv", delimiter=",")
        else:
            nu_value = np.genfromtxt("Samples/Nu-Matrices/Nu-Matrix-Hard-(D).csv", delimiter=",")

dimension_value = probability_values.shape[0]
nu_value = nu_value * n_mean_value

#%% Gillespie Algorithm


total_realisations = 0
current_realisation = 0

while current_realisation < realisations:
    current_state = initial_state[:]
    current_time = 0.0
    while current_time <= time_max:
        r1 = uniform(0.0, 1.0)
        r2 = uniform(0.0, 1.0)
        alpha = rate_list(current_state, probability_values, mu_value, nu_value, dimension_value, stimulus_value, max_level_value)
        alpha_sum = float(sum(alpha))

        dt = -math.log(r1) / alpha_sum
        current_time += dt

        if len(alpha) == 2 * len(current_state):
            for current_rate in range(len(alpha)):
                if (sum(alpha[:current_rate]) / alpha_sum) <= r2 < (sum(alpha[:current_rate + 1]) / alpha_sum):
                    if current_rate % 2 == 0:
                        current_state[int(current_rate / 2)] += 1
                    else:
                        current_state[int(current_rate / 2)] -= 1
                        if current_state.count(0) > 0:
                            break
            else:
                continue
            break
        else:
            for current_rate in range(len(alpha)):
                if (sum(alpha[:current_rate]) / alpha_sum) <= r2 < (sum(alpha[:current_rate + 1]) / alpha_sum):
                    current_state[int(current_rate)] -= 1
                    if current_state.count(0) > 0:
                        break
            else:
                continue
            break
    else:
        if clones == 2:
            distribution[current_state[0] - 1, current_state[1] - 1] += 1
        if clones == 3:
            distribution[current_state[0] - 1, current_state[1] - 1, current_state[2] - 1] += 1
        current_realisation += 1
    total_realisations += 1

#%% Storing results

if clones == 2:
    params = 'Results/QSD/Established/Gillespie/Parameters.bin'
    dat = 'Results/QSD/Established/Gillespie/Data.bin'
if clones == 3:
    if new_clone_is_soft:
        params = 'Results/QSD/Soft/Gillespie/Parameters.bin'
        dat = f'Results/QSD/Soft/Gillespie/Data-{sample_value}.bin'
    else:
        params = 'Results/QSD/Hard/Gillespie/Parameters.bin'
        dat = f'Results/QSD/Hard/Gillespie/Data-{sample_value}.bin'

os.makedirs(os.path.dirname(params), exist_ok=True)
with open(params, 'wb') as file:
    parameters = (["dimension_value", "max_level_value", "mu_value", "gamma_value", "stimulus_value"], dimension_value, max_level_value, mu_value, gamma_value, stimulus_value)
    pickle.dump(parameters, file)

os.makedirs(os.path.dirname(dat), exist_ok=True)
with open(dat, 'wb') as file:
    pickle.dump(distribution, file)
