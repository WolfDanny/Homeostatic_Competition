#%% Packages


from scipy.special import comb
from scipy.stats import uniform
from scipy.sparse import coo_matrix, identity, csc_matrix
from itertools import chain, combinations
import numpy as np
import pickle
import gc
import os

#%% Functions


def clone_sets(dimension, clone):
    """
    Creates an ordered list of tuples representing all subsets of a set of *dimension* elements that include the *clone*-th element.

    Parameters
    ----------
    dimension : int
        Number of elements.
    clone : int
        Specified element (starts at 0).

    Returns
    -------
    List
        list of tuples representing all subsets of a set of dimension elements that include the clone-th element.
    """

    if clone >= dimension or clone < 0:
        return -1

    x = range(dimension)
    sets = list(chain(*[combinations(x, ni) for ni in range(dimension + 1)]))
    d = []

    for T in sets:
        if clone not in T:
            d.insert(0, sets.index(T))

    for i in d:
        sets.pop(i)

    return sets


def level_position(level, dimension, state):
    """
    Calculates the position of *state* in *level*

    Parameters
    ----------
    level : int
        Level of the state space.
    dimension : int
        Number of clonotypes.
    state : list[int]
        List of number of cells per clonotype.

    Returns
    -------
    int
        Position of state in level, or -1 if state is not in level.
    """

    level += dimension
    state = [clone + 1 for clone in state]

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


def level_states(level, dimension):
    """
    Creates a list of all states in *level*.

    Parameters
    ----------
    level : int
        Level of the state space.
    dimension : int
        Number of clonotypes.

    Returns
    -------
    state_list : list
        List of all states in level.
    """

    state_list = []
    n = [0 for _ in range(dimension)]

    while True:

        if len(n) == dimension and sum(n) == level:
            state_list.append(n[:])

        n[0] += 1
        for i in range(len(n)):
            if n[i] > level:
                if (i + 1) < len(n):
                    n[i+1] += 1
                    n[i] = 0
                for j in range(i):
                    n[j] = 0

        if n[-1] > level:
            break

    return state_list


def sum_clones(subset, state):
    """
    Sums the number of cells in clones belonging to *subset* for *state*.

    Parameters
    ----------
    subset : tuple
        Clonotypes in the subset.
    state : list[int]
        Number of cells per clonotype.

    Returns
    -------
    total_cells : float
        Total number of cells in subset for state.
    """

    total_cells = 0.0

    for s in subset:
        total_cells += float(state[s])

    return float(total_cells)


def birth_rate(state, probability, clone, dimension, nu, stimulus):
    """
    Calculates the birth rate for *clone* in *state*.

    Parameters
    ----------
    state : list[int]
        Number of cells per clonotype.
    probability : numpy.ndarray
        Probability matrix.
    clone : int
        Specified clone.
    dimension : int
        Number of clonotypes.
    nu : numpy.ndarray
        Niche overlap matrix.
    stimulus : list[float]
        Stimulus parameters.

    Returns
    -------
    float
        Birth rate for clone in state.
    """

    rate = 0.0
    sets = clone_sets(dimension, clone)

    for i in range(len(sets)):
        if sum_clones(sets[i], state) != 0:
            rate += probability[clone][i] / (sum_clones(sets[i], state) + nu[clone][i])

    return rate * state[clone] * stimulus[clone]


def delta(state, probability, mu, dimension, nu, stimulus):
    """
    Calculates the sum of all birth and death rates for *state*.

    Parameters
    ----------
    state : list[int]
        Number of cells per clonotype.
    probability : list[float]
        Probability matrix.
    mu : float
        Single cell death rate.
    dimension : int
        Number of clonotypes.
    nu : numpy.ndarray
        Niche overlap matrix.
    stimulus : list[float]
        Stimulus parameters.

    Returns
    -------
    delta_value : float
        Sum of all birth and death rates for state.
    """

    delta_value = 0.0

    for i in range(len(state)):
        delta_value += state[i] * mu

    for i in range(len(state)):
        delta_value += birth_rate(state, probability, i, dimension, nu, stimulus)

    return delta_value


def death_delta(state, mu):
    """
    Calculates the sum of all death rates for *state*.

    Parameters
    ----------
    state : list[int]
        Number of cells per clonotype.
    mu : float
        Single cell death rate.

    Returns
    -------
    delta_value : float
        Sum of all death rates for state.
    """

    delta_value = 0.0

    for i in range(len(state)):
        delta_value += state[i] * mu

    return delta_value


def death_diagonal_matrices(level, max_level, dimension, probability, stimulus, mu, nu):
    """
    Creates the subdiagonal matrix A_{level, level - 1}.

    Parameters
    ----------
    level : int
        Level in the state space.
    max_level : int
        Maximum level of the state space.
    dimension : int
        Number of clonotypes.
    probability : numpy.ndarray
        Probability matrix.
    stimulus : list[float]
        Stimulus parameters.
    mu : float
        Single cell death rate.
    nu : numpy.ndarray
        Niche overlap matrix.

    Returns
    -------
    dd_matrix : csc_matrix
        Matrix A_{level, level - 1}.
    """

    rows = []
    cols = []
    data = []

    matrix_shape = (int(comb(level + dimension - 1, dimension - 1)), int(comb(level + dimension - 2, dimension - 1)))

    states = level_states(level, dimension)

    if level < max_level:
        for state in states:
            for i in range(len(state)):
                new_state = state[:]
                new_state[i] -= 1
                if new_state.count(-1) == 0:
                    data.append((state[i] * mu) / delta(state, probability, mu, dimension, nu, stimulus))
                    cols.append(level_position(level - 1, dimension, new_state))
                    rows.append(level_position(level, dimension, state))
    else:
        for state in states:
            for i in range(len(state)):
                new_state = state[:]
                new_state[i] -= 1
                if new_state.count(-1) == 0:
                    data.append((state[i] * mu) / death_delta(state, mu))
                    cols.append(level_position(level - 1, dimension, new_state))
                    rows.append(level_position(level, dimension, state))

    dd_matrix = coo_matrix((data, (rows, cols)), matrix_shape).tocsc()

    return dd_matrix


def birth_diagonal_matrices(level, clone, dimension, probability, stimulus, mu, nu):
    """
    Creates the superdiagonal matrix A^{clone}_{level, level + 1}.

    Parameters
    ----------
    level : int
        Level in the state space.
    clone : int
        Dividing clonotype.
    dimension : int
        Number of clonotypes.
    probability : numpy.ndarray
        Probability matrix.
    stimulus : list[float]
        Stimulus parameters.
    mu : float
        Single cell death rate.
    nu : numpy.ndarray
        Niche overlap matrix.

    Returns
    -------
    bd_matrix : csc_matrix
        Matrix A_{level, level + 1}.
    """

    rows = []
    cols = []
    data = []

    matrix_shape = (int(comb(level + dimension - 1, dimension - 1)), int(comb(level + dimension, dimension - 1)))

    states = level_states(level, dimension)

    for state in states:
        if state[clone] != 0:  # state.count(0) == 0:
            for i in range(len(state)):
                if i != clone:
                    new_state = state[:]
                    new_state[i] += 1

                    cols.append(level_position(level + 1, dimension, new_state))
                    rows.append(level_position(level, dimension, state))
                    try:
                        data.append(birth_rate(state, probability, i, dimension, nu, stimulus) / delta(state, probability, mu, dimension, nu, stimulus))
                    except ZeroDivisionError:
                        print(level)
                        if level == 0:
                            cols.pop()
                            rows.pop()

    bd_matrix = coo_matrix((data, (rows, cols)), matrix_shape).tocsc()

    return bd_matrix


def division_vector(level, clone, divisions, max_level, dimension, probability, stimulus, mu, nu, probability_previous_division=None):
    """
    Creates the division vector d^{(*clone*)}_{*divisions*, *level*} as a csc_matrix.

    Parameters
    ----------
    level : int
        Level in the state space.
    clone : int
        Dividing clonotype.
    divisions : int
        Number of divisions.
    max_level : int
        Maximum level of the state space.
    dimension : int
        Number of clonotypes.
    probability : numpy.ndarray
        Probability matrix.
    stimulus : list[float]
        Stimulus parameters.
    mu : float
        Single cell death rate.
    nu : numpy.ndarray
        Niche overlap matrix.
    probability_previous_division : list[csc_matrix]
        Division probability vector for (divisions - 1), for divisions = 0 this parameter is not required.

    Returns
    -------
    d_vector : csc_matrix
        Division vector d^{(*clone*)}_{*divisions*, *level*} as a csc_matrix.
    """

    if probability_previous_division is None:
        probability_previous_division = []

    try:
        probability_previous_division = probability_previous_division[level].todense().flatten().tolist()[0]
    except IndexError:
        if divisions == 0 or level == 0:
            pass
        else:
            return -1

    rows = []
    cols = []
    data = []

    states = level_states(level, dimension)

    matrix_shape = (int(comb(level + dimension - 1, dimension - 1)), 1)

    if level != 0:
        if divisions != 0:
            if level < max_level:
                for state in states:
                    rows.append(level_position(level, dimension, state))
                    cols.append(0)
                    data.append((birth_rate(state, probability, clone, dimension, nu, stimulus) / delta(state, probability, mu, dimension, nu, stimulus)) * probability_previous_division[level_position(level, dimension, state)])
        else:
            for state in states:
                if state[clone] == 0:
                    rows.append(level_position(level, dimension, state))
                    cols.append(0)
                    data.append(1)
                elif level < max_level and state[clone] == 1:
                    rows.append(level_position(level, dimension, state))
                    cols.append(0)
                    data.append((state[clone] * mu) / delta(state, probability, mu, dimension, nu, stimulus))
    else:
        if divisions == 0:
            rows.append(0)
            cols.append(0)
            data.append(1)

    d_vector = coo_matrix((data, (rows, cols)), matrix_shape).tocsc()
    d_vector.eliminate_zeros()

    return d_vector

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
    b_matrices.append(birth_diagonal_matrices(level_value, dividing_clone, dimension_value, probability_values, stimulus_value, mu_value, nu_value))

# Calculating lower diagonal (death) matrices, and storing them in order
for level_value in range(1, max_level_value + 1):
    d_matrices.append(death_diagonal_matrices(level_value, max_level_value, dimension_value, probability_values, stimulus_value, mu_value, nu_value))

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
