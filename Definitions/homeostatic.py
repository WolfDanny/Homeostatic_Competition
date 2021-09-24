from scipy.special import comb
from scipy.sparse import coo_matrix, identity, csc_matrix, dok_matrix
from itertools import chain, combinations
import numpy as np
import pickle
import gc
import os


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
