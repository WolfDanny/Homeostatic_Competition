#%% Packages


from scipy.special import comb
from scipy.stats import uniform
from itertools import chain, combinations
from copy import deepcopy
from scipy.sparse import coo_matrix, csc_matrix
from scipy.sparse.linalg import inv
import numpy as np
import random
import pickle
import math
import tikzplotlib
import seaborn as sns
import matplotlib.pyplot as plt

#%% Global parameters


new_clone_is_soft = False
max_level_value = 179
mu_value = 1.0
n_mean_value = 10
gamma_value = 1.0
stimulus_value = [8 * gamma_value, 8 * gamma_value, 8 * gamma_value]
sample_value = 0
nu_value = np.asarray([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])

realisations = 1
time_max = 100
initial_state = [15, 15, 15]

#%% Reading Samples and variables [Paper results]


probability_values = np.genfromtxt("../Samples/Matrices/Matrix-{}.csv".format(0), delimiter=",")
dimension_value = probability_values.shape[0]

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


def probability_matrix(dimension, stimulus, sample):
    """
    Creates the probability matrix from *stimulus* and *sample*.

    Parameters
    ----------
    dimension : int
        Number of clonotypes.
    stimulus : List[int]
        Stimulus parameters.
    sample : List[float]
        List of probability values sampled.

    Returns
    -------
    p_matrix : numpy.ndarray
        List expression of the probability matrix.
    """

    sets = []
    set_matrix = [[0 for _ in range(2 ** (dimension - 1))] for _ in range(dimension)]
    sample_matrix = [[None for _ in range(2 ** (dimension - 1))] for _ in range(dimension)]
    p_matrix = np.zeros(shape=(dimension, 2 ** (dimension - 1)))

    # Creating set matrix
    for i in range(dimension):
        sets.append(clone_sets(dimension, i))

    for row in range(len(set_matrix)):
        for col in range(len(set_matrix[row])):
            for i in range(len(set_matrix)):
                for j in range(len(set_matrix[i])):
                    if set_matrix[row][col] == 0 and sets[row][col] == sets[i][j]:
                        set_matrix[row][col] = [i, j]
                        break
                else:
                    continue
                break
    del sets

    # Creating sample matrix
    sample_pos = 0
    for row in range(len(set_matrix)):
        for col in range(1, len(set_matrix[row])):
            if set_matrix[row][-col] == [row, len(set_matrix[row]) - col]:
                sample_matrix[row][-col] = sample_pos
                for i in [x for x in range(len(set_matrix)) if x != row]:
                    try:
                        sample_matrix[i][set_matrix[i].index(set_matrix[row][-col])] = sample_pos
                    except ValueError:
                        pass
                sample_pos += 1

    # Creating probability matrix
    max_sample = 0
    for pos, sample_number in reversed(list(enumerate(sample_matrix[0]))):
        if sample_number is not None:
            p_matrix[0][pos] = uniform(loc=0, scale=1 - p_matrix[0].sum()).ppf(sample[sample_number])
            for row in range(1, p_matrix.shape[0]):
                try:
                    position = sample_matrix[row].index(sample_number)
                    p_matrix[row][position] = p_matrix[0][pos] * (stimulus[0] / stimulus[row])
                except ValueError:
                    pass
            if sample_number > max_sample:
                max_sample = sample_number
        else:
            p_matrix[0][pos] = 1 - p_matrix[0].sum()

    for sample_number in range(max_sample + 1, len(sample)):
        rows = []
        row_totals = []
        for row in range(1, p_matrix.shape[0]):
            if sample_number in sample_matrix[row]:
                rows.append(row)
                row_totals.append(p_matrix[row].sum())
        for row in rows:
            current_value = uniform(loc=0, scale=1 - row_totals[rows.index(row)]).ppf(sample[sample_number])
            remaining_values = []
            for row_test in [x for x in rows if x != row]:
                if row_totals[rows.index(row_test)] + (current_value * (stimulus[row] / stimulus[row_test])) > 1:
                    break
                else:
                    remaining_values.append(current_value * (stimulus[row] / stimulus[row_test]))
            else:
                p_matrix[row][sample_matrix[row].index(sample_number)] = current_value
                remaining_values = remaining_values[::-1]
                for remaining_rows in [x for x in rows if x != row]:
                    p_matrix[remaining_rows][sample_matrix[remaining_rows].index(sample_number)] = remaining_values.pop()
                break

    for row in range(1, p_matrix.shape[0]):
        p_matrix[row][0] = 1 - p_matrix[row].sum()

    return p_matrix


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


def death_rate(state, clone, mu, model):
    """
    Calculates the death rate for *clone* in *state* in *model*.

    Parameters
    ----------
    state : List[int]
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


def delta(state, probability, mu, dimension, nu, stimulus, model):
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


def death_delta(state, mu, model):
    """
    Calculates the sum of all death rates for *state*.

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


def rate_list(state, probability, dimension, nu, mu, stimulus):

    rate_list_values = []

    for clone in range(dimension):
        rate_list_values.append(float(birth_rate(state, probability, clone, dimension, nu, stimulus)))
        rate_list_values.append(state[clone] * mu)

    return np.array(rate_list_values)


#%% Linear level reduction algorithm

clone_states = [[] for _ in range(realisations)]
times = [[] for _ in range(realisations)]

for current_realisation in range(realisations):

    current_state = deepcopy(initial_state)
    current_time = 0.0

    while current_time <= time_max:

        if current_state.count(0)== dimension_value:
            break

        r1 = random.uniform(0.0, 1.0)
        r2 = random.uniform(0.0, 1.0)
        alpha = rate_list(current_state, probability_values, dimension_value, nu_value, mu_value, stimulus_value)

        alpha_sum = alpha.sum()

        dt = -math.log(r1) / alpha_sum
        current_time += dt

        for current_rate in range(alpha.size):
            if (sum(alpha[:current_rate]) / alpha_sum) <= r2 < (sum(alpha[:current_rate + 1]) / alpha_sum):
                current_clone = int(current_rate / 2)
                if current_rate % 2 == 0:
                    current_state[current_clone] += 1
                else:
                    current_state[current_clone] -= 1
                clone_states[current_realisation].append(deepcopy(current_state))
                times[current_realisation].append(deepcopy(current_time))

with open('Data.bin', 'wb') as file:
    data = (clone_states, times, realisations)
    pickle.dump(data, file)
