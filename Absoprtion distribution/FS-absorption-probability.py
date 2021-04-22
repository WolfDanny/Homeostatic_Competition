#%% Packages


from scipy.special import comb
from scipy.sparse import coo_matrix, identity, csc_matrix, dok_matrix
from itertools import chain, combinations
import numpy as np
import pickle
import gc
import os

#%% Global parameters


new_clone_is_soft = False
mu_value = 1.0
n_mean_value = 10
gamma_value = 1.0
stimulus_value = [10 * gamma_value, 10 * gamma_value, 10 * gamma_value]

#%% Reading Samples and Variables [Paper results]


SampleHolder = 0
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

with open('../Truncated_levels.bin', 'rb') as file:
    truncated_levels = np.array(pickle.load(file))
    truncated_levels = [max(truncated_levels[:, i]) for i in range(truncated_levels.shape[1])]

max_level_value = max(truncated_levels) + 15  # truncated_levels[SampleHolder]

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


def level_states(level, dimension):
    """
    Creates a list of all non-absorbed states in *level*.

    Parameters
    ----------
    level : int
        Level of the state space.
    dimension : int
        Number of clonotypes.

    Returns
    -------
    state_list : List
        List of all states in level.
    """

    state_list = []
    n = [1 for _ in range(dimension)]

    while True:

        if len(n) == dimension and sum(n) == level and (n.count(0) == 0):
            state_list.append(n[:])

        n[0] += 1
        for i in range(len(n)):
            if n[i] > level - dimension + 1:
                if (i + 1) < len(n):
                    n[i+1] += 1
                    n[i] = 1
                for j in range(i):
                    n[j] = 1

        if n[-1] > level - dimension + 1:
            break

    return state_list


def sum_clones(subset, state):
    """
    Sums the number of cells in clones belonging to *subset* for *state*.

    Parameters
    ----------
    subset : tuple
        Clonotypes in the subset.
    state : List[int]
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
    state : List[int]
        Number of cells per clonotype.
    probability : numpy.ndarray
        Probability matrix.
    clone : int
        Specified clone.
    dimension : int
        Number of clonotypes.
    nu : numpy.ndarray
        Niche overlap matrix.
    stimulus : List[float]
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
    state : List[int]
        Number of cells per clonotype.
    probability : List[float]
        Probability matrix.
    mu : float
        Single cell death rate.
    dimension : int
        Number of clonotypes.
    nu : numpy.ndarray
        Niche overlap matrix.
    stimulus : List[float]
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
    state : List[int]
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
    Creates the sub-diagonal matrix A_{level, level - 1}.

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
    stimulus : List[float]
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

    matrix_shape = (int(comb(level - 1, dimension - 1)), int(comb(level - 2, dimension - 1)))

    states = level_states(level, dimension)

    if level < max_level:
        for state in states:
            for i in range(len(state)):
                new_state = state[:]
                new_state[i] -= 1
                if new_state.count(0) == 0:
                    data.append((state[i] * mu) / delta(state, probability, mu, dimension, nu, stimulus))
                    cols.append(level_position(level - 1, dimension, new_state))
                    rows.append(level_position(level, dimension, state))
    else:
        for state in states:
            for i in range(len(state)):
                new_state = state[:]
                new_state[i] -= 1
                if new_state.count(0) == 0:
                    data.append((state[i] * mu) / death_delta(state, mu))
                    cols.append(level_position(level - 1, dimension, new_state))
                    rows.append(level_position(level, dimension, state))

    dd_matrix = coo_matrix((data, (rows, cols)), matrix_shape).tocsc()

    return dd_matrix


def birth_diagonal_matrices(level, dimension, probability, stimulus, mu, nu):
    """
    Creates the diagonal matrix A_{level, level + 1}.

    Parameters
    ----------
    level : int
        Level in the state space.
    dimension : int
        Number of clonotypes.
    probability : numpy.ndarray
        Probability matrix.
    stimulus : List[float]
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

    matrix_shape = (int(comb(level - 1, dimension - 1)), int(comb(level, dimension - 1)))

    states = level_states(level, dimension)

    for state in states:
        for i in range(len(state)):
            new_state = state[:]
            new_state[i] += 1

            data.append(birth_rate(state, probability, i, dimension, nu, stimulus) / delta(state, probability, mu, dimension, nu, stimulus))
            cols.append(level_position(level + 1, dimension, new_state))
            rows.append(level_position(level, dimension, state))

    bd_matrix = coo_matrix((data, (rows, cols)), matrix_shape).tocsc()

    return bd_matrix


def absorption_matrix(level, clone, max_level, dimension, mu, nu, probability, stimulus):
    """
    Creates the transition matrix R^{*clone*}_{*level*, *level* - 1} in the embedded Markov chain as a csc_matrix.

    Parameters
    ----------
    level : int
        Level in the state space.
    clone : int
        Clone checked for extinction.
    max_level : int
        Maximum level of the state space.
    dimension : int
        Number of clonotypes.
    mu : float
        Single cell death rate.
    nu : numpy.ndarray
        Niche overlap matrix.
    probability : numpy.ndarray
        Probability matrix.
    stimulus : List[float]
        Stimulus parameters.

    Returns
    -------
    a_matrix : csc_matrix
        Transition matrix R^{*clone*}_{*level*, *level - 1*} in the embedded Markov process.
    """

    rows = []
    cols = []
    data = []

    matrix_shape = (int(comb(level - 1, dimension - 1)), int(comb(level - 2, dimension - 2)))

    states = level_states(level, dimension)

    if level < max_level:
        for state in states:
            for i in range(len(state)):
                new_state = state[:]
                new_state[i] -= 1
                if new_state.count(0) == 1 and new_state.index(0) == clone:
                    projection = new_state[:]
                    projection.pop(projection.index(0))

                    data.append((state[i] * mu) / delta(state, probability, mu, dimension, nu, stimulus))
                    cols.append(level_position(level - 1, dimension - 1, projection))
                    rows.append(level_position(level, dimension, state))
    else:
        for state in states:
            for i in range(len(state)):
                new_state = state[:]
                new_state[i] -= 1
                if new_state.count(0) == 1 and new_state.index(0) == clone:
                    projection = new_state[:]
                    projection.pop(projection.index(0))

                    data.append((state[i] * mu) / death_delta(state, mu))
                    cols.append(level_position(level - 1, dimension - 1, projection))
                    rows.append(level_position(level, dimension, state))

    a_matrix = coo_matrix((data, (rows, cols)), matrix_shape).tocsc()

    return a_matrix

#%% Solving the matrix equations


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

if new_clone_is_soft:
    folder = 'Soft'
else:
    folder = 'Hard'

filename = '../Results/Absorption distribution/{0}/Parameters-{1}.bin'.format(folder, SampleHolder)
os.makedirs(os.path.dirname(filename), exist_ok=True)
with open(filename, 'wb') as file:
    parameters = (["dimension_value", "max_level_value", "mu_value", "gamma_value", "stimulus_value"], dimension_value, max_level_value, mu_value, gamma_value, stimulus_value)
    pickle.dump(parameters, file)

filename = '../Results/Absorption distribution/{0}/Data-{1}.bin'.format(folder, SampleHolder)
os.makedirs(os.path.dirname(filename), exist_ok=True)
with open(filename, 'wb') as file:
    pickle.dump(distribution, file)
