#%% Packages


from homeostatic import *
from scipy.special import comb
from scipy.sparse import identity, csc_matrix, dok_matrix
import numpy as np
import pickle
import gc
import os

#%% Parameters


new_clone_is_soft = False
mu_value = 1.0
n_mean_value = 10
gamma_value = 1.0
stimulus_value = [10 * gamma_value, 10 * gamma_value, 10 * gamma_value]
sample_value = 0

#%% Reading Samples and Variables


probability_values = np.genfromtxt(f"Samples/Matrices/Matrix-{sample_value}.csv", delimiter=",")
dimension_value = probability_values.shape[0]

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
nu_value = nu_value * n_mean_value

with open('Results/QSD/Truncated_levels.bin', 'rb') as file:
    truncated_levels = np.array(pickle.load(file))

niche = 0
if new_clone_is_soft:
    niche = 1
max_level_value = max([max(truncated_levels[niche, :, i]) for i in range(truncated_levels.shape[2])]) + 15

#%% Solving matrix equations


b_matrices = []  # Lis of upper diagonal (birth) matrices
d_matrices = []  # List of lower diagonal (death) matrices
a_matrices = [[] for _ in range(dimension_value)]  # List of absorption matrices
distribution = [[] for _ in range(dimension_value)]  # Distribution of absorption matrices

# Calculating upper diagonal (birth) matrices
for level_value in range(dimension_value, max_level_value):
    b_matrices.append(birth_diagonal_matrices(level_value, dimension_value, probability_values, stimulus_value, mu_value, nu_value))

# Calculating lower diagonal (death) matrices
for level_value in range(dimension_value + 1, max_level_value + 1):
    d_matrices.append(death_diagonal_matrices(level_value, max_level_value, dimension_value, probability_values, stimulus_value, mu_value, nu_value))

# Calculating absorption matrices, all zero matrices are stored too
for clone_number in range(dimension_value):
    for absorbing_level_value in range(dimension_value - 1, max_level_value):
        block_column = []
        for level_value in range(dimension_value, max_level_value + 1):
            if absorbing_level_value != level_value - 1:
                block_column.append(dok_matrix((int(comb(level_value - 1, dimension_value - 1)), int(comb(absorbing_level_value - 1, dimension_value - 2)))).tocsc())
            else:
                block_column.append(absorption_matrix(level_value, clone_number, max_level_value, dimension_value, mu_value, nu_value, probability_values, stimulus_value))
        a_matrices[clone_number].append(block_column)

# Calculating the inverses of H matrices, and storing them in inverse order
h_matrices = [identity(d_matrices[-1].shape[0], format="csc")]

for level_order in range(len(d_matrices)):
    gc.collect()
    matrix = identity(b_matrices[-(level_order + 1)].shape[0], format="csc") - b_matrices[-(level_order + 1)].dot(h_matrices[-1].dot(d_matrices[-(level_order + 1)]))
    matrix = np.linalg.inv(matrix.todense())
    h_matrices.append(csc_matrix(matrix))

for clone_number in range(dimension_value):
    for column_number in range(len(a_matrices[clone_number])):
        # Calculating K matrices for the *column_number* column, and storing them in inverse order
        k_matrices = [a_matrices[clone_number][column_number][-1]]
        for level_order in range(len(a_matrices[clone_number][column_number]) - 1):
            k_matrices.append(b_matrices[-(level_order + 1)].dot(h_matrices[level_order].dot(k_matrices[-1])) + a_matrices[clone_number][column_number][-(level_order + 2)])

        # Calculating the distribution of absorption sub-matrices for the *column_number* column
        distribution_column = [h_matrices[-1].dot(k_matrices[-1])]
        for level_order in range(len(k_matrices) - 1):
            matrix_term = d_matrices[level_order].dot(distribution_column[-1]) + k_matrices[-(level_order + 2)]
            distribution_column.append(h_matrices[-(level_order + 2)].dot(matrix_term))
        distribution[clone_number].append(distribution_column)

#%% Storing Data

folder = 'Hard'
if new_clone_is_soft:
    folder = 'Soft'

filename = f'Results/Absorption distribution/{folder}/Parameters-{sample_value}.bin'
os.makedirs(os.path.dirname(filename), exist_ok=True)
with open(filename, 'wb') as file:
    parameters = (["dimension_value", "max_level_value", "mu_value", "gamma_value", "stimulus_value"], dimension_value, max_level_value, mu_value, gamma_value, stimulus_value)
    pickle.dump(parameters, file)

filename = f'Results/Absorption distribution/{folder}/Data-{sample_value}.bin'
os.makedirs(os.path.dirname(filename), exist_ok=True)
with open(filename, 'wb') as file:
    pickle.dump(distribution, file)
