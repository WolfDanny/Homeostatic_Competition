import gc
import math
import os
import pickle
from itertools import chain, combinations

import numpy as np
from scipy.sparse import coo_matrix, csc_matrix, dok_matrix, identity
from scipy.special import comb


def clone_sets(dimension, clone):
    """
    Creates an ordered list of tuples representing all subsets of a set of ``dimension`` elements that include the ``clone``-th element.

    Parameters
    ----------
    dimension : int
        Number of elements.
    clone : int
        Specified element (starts at 0).

    Returns
    -------
    list[tuple[int]]
        List of tuples representing all subsets of a set of dimension elements that include the clone-th element.
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
    Calculates the position of ``state`` in ``level``.

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
        Position of state in level, or -1 if state is not in the level.
    """

    if level == dimension and state.count(1) == dimension:
        return 0

    if len(state) != dimension or sum(state) != level or state.count(0) > 0:
        return -1

    position = 0

    max_cells = level - dimension + 1

    for i in range(dimension):
        position += (state[i] - 1) * (max_cells**i)

    for i in range(dimension - 2):
        position += (state[dimension - 1 - i] - 1) * (
            1 - (max_cells ** (dimension - 1 - i))
        )

    position = int(position / (max_cells - 1))

    for i in range(dimension - 2):
        position += int(
            comb(level - 1 - sum(state[dimension - i : dimension]), dimension - 1 - i)
        ) - int(
            comb(level - sum(state[dimension - i - 1 : dimension]), dimension - 1 - i)
        )

    return int(position - 1)


def level_position_full_space(level, dimension, state):
    """
    Calculates the position of ``state`` + ``(1, ..., 1)`` in ``level`` + ``dimension``.

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
        Position of state in level, or -1 if state is not in the level.
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
        position += (state[i] - 1) * (max_cells**i)

    for i in range(dimension - 2):
        position += (state[dimension - 1 - i] - 1) * (
            1 - (max_cells ** (dimension - 1 - i))
        )

    position = int(position / (max_cells - 1))

    for i in range(dimension - 2):
        position += int(
            comb(level - 1 - sum(state[dimension - i : dimension]), dimension - 1 - i)
        ) - int(
            comb(level - sum(state[dimension - i - 1 : dimension]), dimension - 1 - i)
        )

    return int(position - 1)


def level_states(level, dimension):
    """
    Creates a list of all non-absorbed states in ``level``.

    Parameters
    ----------
    level : int
        Level of the state space.
    dimension : int
        Number of clonotypes.

    Returns
    -------
    state_list : list[list[int]]
        List of all states in level.
    """

    state_list = []
    n = [1 for _ in range(dimension)]

    while True:

        if sum(n) == level:
            state_list.append(n[:])

        n[0] += 1
        for i, _ in enumerate(n):
            if n[i] > level - dimension + 1:
                if (i + 1) < len(n):
                    n[i + 1] += 1
                    n[i] = 1
                for j in range(i):
                    n[j] = 1

        if n[-1] > level - dimension + 1:
            break

    return state_list


def level_states_full_space(level, dimension):
    """
    Creates a list of all states in ``level``.

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

        if sum(n) == level:
            state_list.append(n[:])

        n[0] += 1
        for i, _ in enumerate(n):
            if n[i] > level:
                if (i + 1) < len(n):
                    n[i + 1] += 1
                    n[i] = 0
                for j in range(i):
                    n[j] = 0

        if n[-1] > level:
            break

    return state_list


def sum_clones(subset, state):
    """
    Sums the number of cells in clones belonging to ``subset`` for ``state``.

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
    Calculates the birth rate for ``clone`` in ``state``.

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

    for i, current_set in enumerate(sets):
        if sum_clones(current_set, state) != 0:
            rate += probability[clone][i] / (
                sum_clones(current_set, state) + nu[clone][i]
            )

    return rate * state[clone] * stimulus[clone]


def death_rate(state, clone, mu, model):
    """
    Calculates the death rate for ``clone`` in ``state`` in the approximating process X^{``model``}.

    Parameters
    ----------
    state : list[int]
        Number of cells per clonotype.
    clone : int
        Specified clone.
    mu : float
        Single cell death rate.
    model : int
        Auxiliary process (0 = X^1, 1 = X^2).

    Returns
    -------
    float
        Death rate for clone in state.
    """

    if model == 1:
        if state[clone] > 1:
            return state[clone] * mu
        else:
            return 0.0
    if model == 2:
        return (state[clone] - 1) * mu


def rate_list(state, probability, mu, nu, dimension, stimulus, max_level):
    """
    Creates a list of the birth and death rate of all clonotypes in order.

    If the state in on the maximum level creates a list of the death rate of all clonotypes in order.

    Parameters
    ----------
    state : list[int]
        Number of cells per clonotype.
    probability : numpy.ndarray
        Probability matrix.
    mu : float
        Single cell death rate.
    nu : numpy.ndarray
        Niche overlap matrix.
    dimension : int
        Number of clonotypes.
    stimulus : list[float]
        Stimulus parameters.
    max_level : int
        Maximum level of the state space.

    Returns
    -------
    rates : list[float]
        List of all birth and death rates for all clonotypes.

    """

    rates = []

    if sum(state) < max_level:
        for i in range(dimension):
            rates.append(
                float(birth_rate(state, probability, i, dimension, nu, stimulus))
            )
            rates.append(float(state[i] * mu))
    else:
        for i in range(dimension):
            rates.append(float(state[i] * mu))

    return rates


def delta(state, probability, mu, dimension, nu, stimulus):
    """
    Calculates the sum of all birth and death rates for ``state``.

    Parameters
    ----------
    state : list[int]
        Number of cells per clonotype.
    probability : numpy.ndarray
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

    for current_clone in state:
        delta_value += current_clone * mu

    for i, _ in enumerate(state):
        delta_value += birth_rate(state, probability, i, dimension, nu, stimulus)

    return delta_value


def delta_approximation(state, probability, mu, dimension, nu, stimulus, model):
    """
    Calculates the sum of all birth and death rates for *state* in the approximating process X^{``model``}.

    Parameters
    ----------
    state : list[int]
        Number of cells per clonotype.
    probability : numpy.ndarray
        Probability matrix.
    mu : float
        Single cell death rate.
    dimension : int
        Number of clonotypes.
    nu : numpy.ndarray
        Niche overlap matrix.
    stimulus : list[float]
        Stimulus parameters.
    model : int
        Auxiliary process (0 = X^1, 1 = X^2).

    Returns
    -------
    delta_value : float
        Sum of all birth and death rates for state.
    """

    delta_value = 0.0

    for i in range(len(state)):
        delta_value += death_rate(state, i, mu, model)

    for i in range(len(state)):
        delta_value += birth_rate(state, probability, i, dimension, nu, stimulus)

    return delta_value


def death_delta(state, mu):
    """
    Calculates the sum of all death rates for ``state``.

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

    for current_clone in state:
        delta_value += current_clone * mu

    return delta_value


def death_delta_approximation(state, mu, model):
    """
    Calculates the sum of all death rates for ``state`` in the approximating process X^{``model``}.

    Parameters
    ----------
    state : List[int]
        Number of cells per clonotype.
    mu : float
        Single cell death rate.
    model : int
        Auxiliary process (0 = X^1, 1 = X^2).

    Returns
    -------
    delta_value : float
        Sum of all death rates for state.
    """

    delta_value = 0.0

    for i in range(len(state)):
        delta_value += death_rate(state, i, mu, model)

    return delta_value


def main_diagonal_matrices_approximation(
    level, max_level, dimension, probability, mu, nu, stimulus, model
):
    """
    Creates the diagonal matrix A_{``level``, ``level``} in the approximating process X^{``model``}.

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
    mu : float
        Single cell death rate.
    nu : numpy.ndarray
        Niche overlap matrix.
    stimulus : List[float]
        Stimulus parameters.
    model : int
        Auxiliary process (0 = X^1, 1 = X^2).

    Returns
    -------
    md_matrix : csc_matrix
        Matrix A_{level, level}.
    """

    pos = []
    data = []
    matrix_shape = (
        int(comb(level - 1, dimension - 1)),
        int(comb(level - 1, dimension - 1)),
    )

    states = level_states(level, dimension)

    if level < max_level:
        for state in states:
            data.append(
                -delta_approximation(
                    state, probability, mu, dimension, nu, stimulus, model
                )
            )
            pos.append(level_position(level, dimension, state))
    else:
        for state in states:
            data.append(-death_delta_approximation(state, mu, model))
            pos.append(level_position(level, dimension, state))

    md_matrix = coo_matrix((data, (pos, pos)), matrix_shape).tocsc()

    return md_matrix


def death_diagonal_matrices(level, max_level, dimension, probability, stimulus, mu, nu):
    """
    Creates the sub-diagonal matrix A_{``level``, ``level`` - 1}.

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

    matrix_shape = (
        int(comb(level - 1, dimension - 1)),
        int(comb(level - 2, dimension - 1)),
    )

    states = level_states(level, dimension)

    if level < max_level:
        for state in states:
            for i, _ in enumerate(state):
                new_state = state[:]
                new_state[i] -= 1
                if new_state.count(0) == 0:
                    data.append(
                        (state[i] * mu)
                        / delta(state, probability, mu, dimension, nu, stimulus)
                    )
                    cols.append(level_position(level - 1, dimension, new_state))
                    rows.append(level_position(level, dimension, state))
    else:
        for state in states:
            for i, _ in enumerate(state):
                new_state = state[:]
                new_state[i] -= 1
                if new_state.count(0) == 0:
                    data.append((state[i] * mu) / death_delta(state, mu))
                    cols.append(level_position(level - 1, dimension, new_state))
                    rows.append(level_position(level, dimension, state))

    dd_matrix = coo_matrix((data, (rows, cols)), matrix_shape).tocsc()

    return dd_matrix


def death_diagonal_matrices_division(
    level, max_level, clone, dimension, probability, stimulus, mu, nu
):
    """
    Creates the sub-diagonal matrix A_{``level``, ``level`` - 1}.

    Parameters
    ----------
    level : int
        Level in the state space.
    max_level : int
        Maximum level of the state space.
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
    dd_matrix : csc_matrix
        Matrix A_{level, level - 1}.
    """

    rows = []
    cols = []
    data = []

    matrix_shape = (
        int(comb(level + dimension - 1, dimension - 1)),
        int(comb(level + dimension - 2, dimension - 1)),
    )

    states = level_states_full_space(level, dimension)

    if level < max_level:
        for state in states:
            if state[clone] != 0:
                for i, _ in enumerate(state):
                    new_state = state[:]
                    new_state[i] -= 1
                    if new_state.count(-1) == 0:
                        data.append(
                            (state[i] * mu)
                            / delta(state, probability, mu, dimension, nu, stimulus)
                        )
                        cols.append(
                            level_position_full_space(level - 1, dimension, new_state)
                        )
                        rows.append(level_position_full_space(level, dimension, state))
    else:
        for state in states:
            if state[clone] != 0:
                for i, _ in enumerate(state):
                    new_state = state[:]
                    new_state[i] -= 1
                    if new_state.count(-1) == 0:
                        data.append((state[i] * mu) / death_delta(state, mu))
                        cols.append(
                            level_position_full_space(level - 1, dimension, new_state)
                        )
                        rows.append(level_position_full_space(level, dimension, state))

    dd_matrix = coo_matrix((data, (rows, cols)), matrix_shape).tocsc()

    return dd_matrix


def death_diagonal_matrices_approximation(level, dimension, mu, model):
    """
    Creates the sub-diagonal matrix A_{``level``, ``level`` - 1} in the approximating process X^{``model``}.

    Parameters
    ----------
    level : int
        Level in the state space.
    dimension : int
        Number of clonotypes.
    mu : float
        Single cell death rate.
    model : int
        Auxiliary process (0 = X^1, 1 = X^2).

    Returns
    -------
    dd_matrix : csc_matrix
        Matrix A_{level, level - 1}.
    """

    rows = []
    cols = []
    data = []

    matrix_shape = (
        int(comb(level - 1, dimension - 1)),
        int(comb(level - 2, dimension - 1)),
    )

    states = level_states(level, dimension)

    for state in states:
        for i in range(len(state)):
            new_state = state[:]
            new_state[i] -= 1
            if new_state.count(0) == 0:
                data.append(death_rate(state, i, mu, model))
                cols.append(level_position(level - 1, dimension, new_state))
                rows.append(level_position(level, dimension, state))

    dd_matrix = coo_matrix((data, (rows, cols)), matrix_shape).tocsc()

    return dd_matrix


def birth_diagonal_matrices(level, dimension, probability, stimulus, mu, nu):
    """
    Creates the diagonal matrix A_{``level``, ``level`` + 1}.

    Parameters
    ----------
    level : int
        Level in the state space.
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

    matrix_shape = (
        int(comb(level - 1, dimension - 1)),
        int(comb(level, dimension - 1)),
    )

    states = level_states(level, dimension)

    for state in states:
        for i in range(len(state)):
            new_state = state[:]
            new_state[i] += 1

            data.append(
                birth_rate(state, probability, i, dimension, nu, stimulus)
                / delta(state, probability, mu, dimension, nu, stimulus)
            )
            cols.append(level_position(level + 1, dimension, new_state))
            rows.append(level_position(level, dimension, state))

    bd_matrix = coo_matrix((data, (rows, cols)), matrix_shape).tocsc()

    return bd_matrix


def birth_diagonal_matrices_division(
    level, clone, dimension, probability, stimulus, mu, nu
):
    """
    Creates the superdiagonal matrix A^{``clone``}_{``level``, ``level`` + 1}.

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

    matrix_shape = (
        int(comb(level + dimension - 1, dimension - 1)),
        int(comb(level + dimension, dimension - 1)),
    )

    states = level_states_full_space(level, dimension)

    for state in states:
        if state[clone] != 0:  # state.count(0) == 0:
            for i in range(len(state)):
                if i != clone:
                    new_state = state[:]
                    new_state[i] += 1

                    cols.append(
                        level_position_full_space(level + 1, dimension, new_state)
                    )
                    rows.append(level_position_full_space(level, dimension, state))
                    try:
                        data.append(
                            birth_rate(state, probability, i, dimension, nu, stimulus)
                            / delta(state, probability, mu, dimension, nu, stimulus)
                        )
                    except ZeroDivisionError:
                        print(f"Error in {level}, {state}")
                        if level == 0:
                            cols.pop()
                            rows.pop()

    bd_matrix = coo_matrix((data, (rows, cols)), matrix_shape).tocsc()

    return bd_matrix


def birth_diagonal_matrices_approximation(level, dimension, probability, nu, stimulus):
    """
    Creates the diagonal matrix A_{``level``, ``level`` + 1} in the approximating process X^{``model``}.

    Parameters
    ----------
    level : int
        Level in the state space.
    dimension : int
        Number of clonotypes.
    probability : numpy.ndarray
        Probability matrix.
    nu : numpy.ndarray
        Niche overlap matrix.
    stimulus : list[float]
        Stimulus parameters.

    Returns
    -------
    bd_matrix : csc_matrix
        Matrix A_{level, level + 1}.
    """

    rows = []
    cols = []
    data = []

    matrix_shape = (
        int(comb(level - 1, dimension - 1)),
        int(comb(level, dimension - 1)),
    )

    states = level_states(level, dimension)

    for state in states:
        for i in range(len(state)):
            new_state = state[:]
            new_state[i] += 1

            data.append(birth_rate(state, probability, i, dimension, nu, stimulus))
            cols.append(level_position(level + 1, dimension, new_state))
            rows.append(level_position(level, dimension, state))

    bd_matrix = coo_matrix((data, (rows, cols)), matrix_shape).tocsc()

    return bd_matrix


def absorption_matrix(
    level, clone, max_level, dimension, mu, nu, probability, stimulus
):
    """
    Creates the transition matrix R^{``clone``}_{``level``, ``level`` - 1} in the embedded Markov chain as a csc_matrix.

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
    stimulus : list[float]
        Stimulus parameters.

    Returns
    -------
    a_matrix : csc_matrix
        Transition matrix R^{clone}_{level, level - 1} in the embedded Markov process.
    """

    rows = []
    cols = []
    data = []

    matrix_shape = (
        int(comb(level - 1, dimension - 1)),
        int(comb(level - 2, dimension - 2)),
    )

    states = level_states(level, dimension)

    if level < max_level:
        for state in states:
            for i, _ in enumerate(state):
                new_state = state[:]
                new_state[i] -= 1
                if new_state.count(0) == 1 and new_state.index(0) == clone:
                    projection = new_state[:]
                    projection.pop(projection.index(0))

                    data.append(
                        (state[i] * mu)
                        / delta(state, probability, mu, dimension, nu, stimulus)
                    )
                    cols.append(level_position(level - 1, dimension - 1, projection))
                    rows.append(level_position(level, dimension, state))
    else:
        for state in states:
            for i, _ in enumerate(state):
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


def division_vector(
    level,
    clone,
    divisions,
    max_level,
    dimension,
    probability,
    stimulus,
    mu,
    nu,
    probability_previous_division=None,
):
    """
    Creates the division vector d^{(``clone``)}_{``divisions``, ``level``} as a csc_matrix.

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
        Division vector d^{(clone)}_{divisions, level} as a csc_matrix.
    """

    if probability_previous_division is None:
        probability_previous_division = []

    try:
        probability_previous_division = (
            probability_previous_division[level].todense().flatten().tolist()[0]
        )
    except IndexError:
        if divisions == 0 or level == 0:
            pass
        else:
            return -1

    rows = []
    cols = []
    data = []

    states = level_states_full_space(level, dimension)

    matrix_shape = (int(comb(level + dimension - 1, dimension - 1)), 1)

    if level != 0:
        if divisions != 0:
            if level < max_level:
                for state in states:
                    rows.append(level_position_full_space(level, dimension, state))
                    cols.append(0)
                    data.append(
                        (
                            birth_rate(
                                state, probability, clone, dimension, nu, stimulus
                            )
                            / delta(state, probability, mu, dimension, nu, stimulus)
                        )
                        * probability_previous_division[
                            level_position_full_space(level, dimension, state)
                        ]
                    )
        else:
            for state in states:
                if state[clone] == 0:
                    rows.append(level_position_full_space(level, dimension, state))
                    cols.append(0)
                    data.append(1)
                elif level < max_level and state[clone] == 1:
                    rows.append(level_position_full_space(level, dimension, state))
                    cols.append(0)
                    data.append(
                        (state[clone] * mu)
                        / delta(state, probability, mu, dimension, nu, stimulus)
                    )
    else:
        if divisions == 0:
            rows.append(0)
            cols.append(0)
            data.append(1)

    d_vector = coo_matrix((data, (rows, cols)), matrix_shape).tocsc()
    d_vector.eliminate_zeros()

    return d_vector


def coefficient_matrix(probability, max_level, mu, nu, stimulus):
    """
    Creates the coefficient matrix of the difference equations for the mean time to extinction as a csr_matrix.

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
    stimulus : list[float]
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
            for clone, clone_value in enumerate(state):
                new_state = state[:]
                new_state[clone] += 1

                col = next_level + level_position(level + 1, dimension, new_state)
                rows.append(row)
                cols.append(col)
                data.append(
                    birth_rate(state, probability, clone, dimension, nu, stimulus)
                )

                new_state[clone] -= 2

                if new_state.count(0) == 0:
                    col = (
                        previous_level
                        - len(level_states(level - 1, dimension))
                        + level_position(level - 1, dimension, new_state)
                    )
                    rows.append(row)
                    cols.append(col)
                    data.append(clone_value * mu)
    previous_level += len(level_states(max_level - 1, dimension))
    for state in level_states(max_level, dimension):
        row = previous_level + level_position(max_level, dimension, state)
        rows.append(row)
        cols.append(row)
        data.append(-delta(state, probability, mu, dimension, nu, stimulus))
        for clone, clone_value in enumerate(state):
            new_state = state[:]
            new_state[clone] -= 1

            if new_state.count(0) == 0:
                col = (
                    previous_level
                    - len(level_states(max_level - 1, dimension))
                    + level_position(max_level - 1, dimension, new_state)
                )
                rows.append(row)
                cols.append(col)
                data.append(clone_value * mu)

    return coo_matrix(
        (data, (rows, cols)),
        (int(comb(max_level, dimension)), int(comb(max_level, dimension))),
    ).tocsr()


def absorption_distribution(clone, state, dimension, max_level, distribution):
    """
    Extracts the absorption distribution U^{``clone``} for the starting state (``state``) from the complete absorption distribution (``distribution``).

    Parameters
    ----------
    clone : int
        Clonotype becoming extinct.
    state : list[int]
        List of number of cells per clonotype.
    dimension : int
        Number of clonotypes.
    max_level : int
        Maximum level of the state space.
    distribution : list[list[list[csc_matrix]]]
        List of the absorption distributions indexed by clonotype, absorbing level, starting level, and starting state.

    Returns
    -------
    absorption : numpy.ndarray
        Absorption distribution U^{clone} for the starting state.
    """

    absorption_list = [
        distribution[clone][i][sum(state) - dimension]
        .todense()
        .tolist()[level_position(sum(state), dimension, state)]
        for i in range(max_level - 2)
    ]

    absorption = np.zeros((max_level - 2, max_level - 2))

    for current_level, _ in enumerate(distribution[clone]):
        current_states = level_states(current_level + 2, 2)
        for current_state in current_states:
            absorption[current_state[0] - 1, current_state[1] - 1] += absorption_list[
                current_level
            ][level_position(current_level + 2, 2, current_state)]

    return absorption


def absorption_distribution_value(clone, state, dimension, max_level, distribution):
    """
    Extracts the absorption distribution U^{``clone``} and calculates the probability U^{``clone``} for the starting state ``state`` from the complete absorption distribution (``distribution``).

    Parameters
    ----------
    clone : int
        Clonotype becoming extinct.
    state : list[int]
        List of number of cells per clonotype.
    dimension : int
        Number of clonotypes.
    max_level : int
        Maximum level of the state space.
    distribution : list[list[list[csc_matrix]]]
        List of the absorption distributions indexed by clonotype, absorbing level, starting level, and starting state.

    Returns
    -------
    int
        Absorption probability U^{clone} for the starting state.
    """

    absorption_list = [
        distribution[clone][i][sum(state) - dimension]
        .todense()
        .tolist()[level_position(sum(state), dimension, state)]
        for i in range(max_level - 2)
    ]

    return sum([sum(current_level) for current_level in absorption_list])


def hellinger_distance(distributions):
    """
    Calculates  the Hellinger distance between two three-dimensional distributions.

    Parameters
    ----------
    distributions : tuple[numpy.ndarray]
        Tuple of the two distributions to be compared.

    Returns
    -------
    float
        Hellinger distance between the two distributions.
    """
    shapes = (distributions[0].shape[0], distributions[1].shape[0])
    max_index = np.argmax(shapes)
    distance = 0
    for i in range(shapes[max_index]):
        for j in range(shapes[max_index]):
            for k in range(shapes[max_index]):
                try:
                    distance += (
                        math.sqrt(distributions[0][i][j][k])
                        - math.sqrt(distributions[1][i][j][k])
                    ) ** 2
                except IndexError:
                    distance += math.fabs(distributions[max_index][i][j][k])
    return (1 / math.sqrt(2)) * math.sqrt(distance)
