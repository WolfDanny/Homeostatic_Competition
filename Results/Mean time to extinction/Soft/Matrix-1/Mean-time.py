#%% Packages


from scipy.special import comb
from itertools import chain, combinations
from scipy.sparse import coo_matrix, csc_matrix
from scipy.sparse.linalg import spsolve
import numpy as np
import pickle

#%% Global parameters


new_clone_is_soft = True
max_level_value = 100
mu_value = 1.0
n_mean_value = 10
gamma_value = 1.0
# stimulus_value = [10 * gamma_value, 10 * gamma_value]
stimulus_value = [10 * gamma_value, 10 * gamma_value, 10 * gamma_value]
sample_value = 1

#%% Reading Samples and variables [Paper results]


probability_values = np.genfromtxt("../../Samples/Matrices/Matrix-{}.csv".format(sample_value), delimiter=",")
dimension_value = probability_values.shape[0]

if sample_value < 3:
    if new_clone_is_soft:
        nu_value = np.genfromtxt("../../Samples/Nu-Matrices/Nu-Matrix-Soft.csv", delimiter=",")
    else:
        nu_value = np.genfromtxt("../../Samples/Nu-Matrices/Nu-Matrix-Hard.csv", delimiter=",")
else:
    if new_clone_is_soft:
        nu_value = np.genfromtxt("../../Samples/Nu-Matrices/Nu-Matrix-Soft-(D).csv", delimiter=",")
    else:
        nu_value = np.genfromtxt("../../Samples/Nu-Matrices/Nu-Matrix-Hard-(D).csv", delimiter=",")
nu_value = nu_value * n_mean_value

# probability_values = np.genfromtxt("../Samples/Established-Matrix/Matrix-2C.csv", delimiter=",")
# dimension_value = probability_values.shape[0]
# nu_value = np.genfromtxt("../Samples/Established-Nu-Matrix/Nu-Matrix-2C.csv", delimiter=",")
# nu_value = nu_value * n_mean_value

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
    List[int]
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
    state_list : List[List]
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
    probability : numpy.ndarray
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


def coefficient_matrix(probability, max_level, mu, nu, stimulus):
    """
    Creates the coefficient matrix of the difference equations as a csr_matrix.

    Parameters
    ----------
    probability : numpy.ndarray
        Probability matrix.
    max_level : int
        Maximum level in the state space.
    mu : float
        Death rate of a single cell.
    nu : numpy.array
        Mean niche overlap matrix.
    stimulus : List[float]
        Stimulus vector.

    Returns
    -------
    csc_matrix
        Coefficient matrix for the difference equations.
    """

    rows = []
    cols = []
    data = []

    dimension = probability.shape[0]

    previous_level = 0
    next_level = 1
    for state in level_states(dimension, dimension):
        row = level_position(dimension, dimension, state)
        rows.append(row)
        cols.append(row)
        data.append(-delta(state, probability, mu, dimension, nu, stimulus))
        for clone in range(len(state)):
            new_state = state[:]
            new_state[clone] += 1

            col = next_level + level_position(dimension + 1, dimension, new_state)
            rows.append(row)
            cols.append(col)
            data.append(birth_rate(state, probability, clone, dimension, nu, stimulus))
    for level in range(dimension + 1, max_level):
        previous_level += len(level_states(level - 1, dimension))
        next_level = previous_level + len(level_states(level, dimension))
        for state in level_states(level, dimension):
            row = previous_level + level_position(level, dimension, state)
            rows.append(row)
            cols.append(row)
            data.append(-delta(state, probability, mu, dimension, nu, stimulus))
            for clone in range(len(state)):
                new_state = state[:]
                new_state[clone] += 1

                col = next_level + level_position(level + 1, dimension, new_state)
                rows.append(row)
                cols.append(col)
                data.append(birth_rate(state, probability, clone, dimension, nu, stimulus))

                new_state[clone] -= 2

                if new_state.count(0) == 0:
                    col = previous_level - len(level_states(level - 1, dimension)) + level_position(level - 1, dimension, new_state)
                    rows.append(row)
                    cols.append(col)
                    data.append(state[clone] * mu)
    previous_level += len(level_states(max_level - 1, dimension))
    for state in level_states(max_level, dimension):
        row = previous_level + level_position(max_level, dimension, state)
        rows.append(row)
        cols.append(row)
        data.append(-delta(state, probability, mu, dimension, nu, stimulus))
        for clone in range(len(state)):
            new_state = state[:]
            new_state[clone] -= 1

            if new_state.count(0) == 0:
                col = previous_level - len(level_states(level - 1, dimension)) + level_position(max_level - 1, dimension, new_state)
                rows.append(row)
                cols.append(col)
                data.append(state[clone] * mu)

    return coo_matrix((data, (rows, cols)), (int(comb(max_level_value, dimension_value)), int(comb(max_level_value, dimension_value)))).tocsr()

#%% Solving difference equations


M = coefficient_matrix(probability_values, max_level_value, mu_value, nu_value, stimulus_value)
b = [-1] * int(comb(max_level_value, dimension_value))

Solution = spsolve(M, b)

# Storing Data
with open('Data.bin', 'wb') as file:
    pickle.dump(Solution, file)
