#%% Packages


from homeostatic import *
from scipy.special import comb
from scipy.sparse.linalg import spsolve
import numpy as np
import pickle
import os

#%% Parameters


new_clone_is_soft = False
max_level_value = 100
mu_value = 1.0
n_mean_value = 10
gamma_value = 1.0
clones = 2
sample_value = 0

#%% Reading Samples


if clones == 2:
    stimulus_value = [10 * gamma_value, 10 * gamma_value]

    probability_values = np.genfromtxt("../Samples/Established-Matrix/Matrix-2C.csv", delimiter=",")
    nu_value = np.genfromtxt("../Samples/Established-Nu-Matrix/Nu-Matrix-2C.csv", delimiter=",")

if clones == 3:
    stimulus_value = [10 * gamma_value, 10 * gamma_value, 10 * gamma_value]

    probability_values = np.genfromtxt("../Samples/Matrices/Matrix-{}.csv".format(sample_value), delimiter=",")
    if sample_value < 3:
        if new_clone_is_soft:
            nu_value = np.genfromtxt("../Samples/Nu-Matrices/Nu-Matrix-Soft.csv", delimiter=",")
        else:
            nu_value = np.genfromtxt("../Samples/Nu-Matrices/Nu-Matrix-Hard.csv", delimiter=",")
    else:
        if new_clone_is_soft:
            nu_value = np.genfromtxt("../Samples/Nu-Matrices/Nu-Matrix-Soft-(D).csv", delimiter=",")
        else:
            nu_value = np.genfromtxt("../Samples/Nu-Matrices/Nu-Matrix-Hard-(D).csv", delimiter=",")

dimension_value = probability_values.shape[0]
nu_value = nu_value * n_mean_value

#%% Solving difference equations


M = coefficient_matrix(probability_values, max_level_value, mu_value, nu_value, stimulus_value)
b = [-1] * int(comb(max_level_value, dimension_value))

Solution = spsolve(M, b)

#%% Storing Data


os.makedirs(os.path.dirname('../Results/Mean time to extinction/Data.bin'), exist_ok=True)
with open('../Results/Mean time to extinction/Data.bin', 'wb') as file:
    pickle.dump(Solution, file)

params = [new_clone_is_soft, max_level_value, mu_value, n_mean_value, gamma_value, clones, sample_value, dimension_value, nu_value]
os.makedirs(os.path.dirname('../Results/Mean time to extinction/Parameters.bin'), exist_ok=True)
with open('../Results/Mean time to extinction/Parameters.bin', 'wb') as file:
    pickle.dump(params, file)
