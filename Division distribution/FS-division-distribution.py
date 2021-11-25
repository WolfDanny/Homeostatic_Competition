#%% Packages


from homeostatic import *
from scipy.special import comb
from scipy.stats import uniform
from scipy.sparse import coo_matrix, identity, csc_matrix
from itertools import chain, combinations
import numpy as np
import pickle
import gc
import os

#%% Variables


new_clone_is_soft = False
mu_value = 1.0
n_mean_value = 10
gamma_value = 1.0
stimulus_value = [10 * gamma_value, 10 * gamma_value, 10 * gamma_value]
dividing_clone = 2
num_divisions = 200

#%% Reading Samples [Paper results]


SampleHolder = 3
probability_values = np.genfromtxt("../Samples/Matrices/Matrix-{}.csv".format(SampleHolder), delimiter=",")
dimension_value = probability_values.shape[0]

if SampleHolder < 3:
    if new_clone_is_soft:
        nu_value = np.genfromtxt("../Samples/Nu-Matrices/Nu-Matrix-Soft.csv", delimiter=",")
    else:
        nu_value = np.genfromtxt("../Samples/Nu-Matrices/Nu-Matrix-Hard.csv", delimiter=",")
else:
    if new_clone_is_soft:
        nu_value = np.genfromtxt("../Samples/Nu-Matrices/Nu-Matrix-Soft-(D).csv", delimiter=",")
    else:
        nu_value = np.genfromtxt("../Samples/Nu-Matrices/Nu-Matrix-Hard-(D).csv", delimiter=",")
nu_value = nu_value * n_mean_value

if new_clone_is_soft:
    folder = 'Soft'
else:
    folder = 'Hard'

with open('../Truncated_levels.bin', 'rb') as file:
    truncated_levels = np.array(pickle.load(file))
    truncated_levels = [max(truncated_levels[:, i]) for i in range(truncated_levels.shape[1])]

max_level_value = max(truncated_levels)  # truncated_levels[SampleHolder]
#%% Solving the matrix equations


b_matrices = []  # Lis of upper diagonal (birth) matrices
d_matrices = []  # List of lower diagonal (death) matrices

# Calculating upper diagonal (birth) matrices, and storing them in order
for level_value in range(max_level_value):
    b_matrices.append(birth_diagonal_matrices_division(level_value, dividing_clone, dimension_value, probability_values, stimulus_value, mu_value, nu_value))

# Calculating lower diagonal (death) matrices, and storing them in order
for level_value in range(1, max_level_value + 1):
    d_matrices.append(death_diagonal_matrices_full_space(level_value, max_level_value, dimension_value, probability_values, stimulus_value, mu_value, nu_value))

# Calculating the inverses of H matrices, and storing them in inverse order
h_matrices = [identity(d_matrices[-1].shape[0], format="csc")]

for level_order in range(len(d_matrices)):
    gc.collect()
    matrix = identity(b_matrices[-(level_order + 1)].shape[0], format="csc") - b_matrices[-(level_order + 1)].dot(h_matrices[-1].dot(d_matrices[-(level_order + 1)]))
    matrix_inv = np.linalg.inv(matrix.todense())
    h_matrices.append(csc_matrix(matrix_inv))

# Solving the matrix equation for all division numbers
for current_division in range(num_divisions + 1):
    distribution = []  # Probability of division vectors

    # Calculating division vectors
    d_vectors = []
    if current_division != 0:
        file = open('../Results/Division distribution/{0}/Matrix-{1}/Clone-{2}/Data-{3}.bin'.format(folder, SampleHolder, dividing_clone + 1, current_division - 1), 'rb')
        previous_division = pickle.load(file)
        file.close()
        for current_level in range(max_level_value + 1):
            d_vectors.append(division_vector(current_level, dividing_clone, current_division, max_level_value, dimension_value, probability_values, stimulus_value, mu_value, nu_value, previous_division))
    else:
        for current_level in range(max_level_value + 1):
            d_vectors.append(division_vector(current_level, dividing_clone, current_division, max_level_value, dimension_value, probability_values, stimulus_value, mu_value, nu_value))

    # Calculating the K vectors, and storing them in inverse order
    k_vectors = [d_vectors[-1]]

    for level_order in range(len(b_matrices)):
        vector = b_matrices[-(level_order + 1)].dot(h_matrices[level_order].dot(k_vectors[-1])) + d_vectors[-(level_order + 2)]
        k_vectors.append(vector)

    # Calculating the probabilities of current_division divisions happening before extinction
    if num_divisions != 0:
        distribution.append(coo_matrix(([], ([], [])), [1, 1]).tocsc())
    else:
        distribution.append(coo_matrix(([1], ([0], [0])), [1, 1]).tocsc())

    for level_order in range(max_level_value):
        matrix_term = d_matrices[level_order].dot(distribution[-1]) + k_vectors[-(level_order + 2)]
        distribution_value = h_matrices[-(level_order + 2)].dot(matrix_term)
        distribution.append(distribution_value)

    # Storing current division number results
    filename = '../Results/Division distribution/{0}/Matrix-{1}/Clone-{2}/Data-{3}.bin'.format(folder, SampleHolder, dividing_clone + 1, current_division)
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'wb') as file:
        pickle.dump(distribution, file)

#%% Storing Data

filename = '../Results/Division distribution/{0}/Matrix-{1}/Clone-{2}/Parameters-{3}.bin'.format(folder, SampleHolder, dividing_clone + 1, SampleHolder)
os.makedirs(os.path.dirname(filename), exist_ok=True)
with open(filename, 'wb') as file:
    parameters = (["dimension_value", "max_level_value", "mu_value", "gamma_value", "stimulus_value", "num_divisions"], dimension_value, max_level_value, mu_value, gamma_value, stimulus_value, num_divisions)
    pickle.dump(parameters, file)
