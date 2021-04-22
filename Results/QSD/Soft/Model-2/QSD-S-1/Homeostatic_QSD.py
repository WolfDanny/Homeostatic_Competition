#%% Packages


from scipy.special import comb
from scipy.stats import uniform
from itertools import chain, combinations
from scipy.sparse import coo_matrix, csc_matrix
from scipy.sparse.linalg import inv
import numpy as np
import pickle
import gc

#%% Global parameters


new_clone_is_soft = True
max_level_value = 179
mu_value = 1.0
n_mean_value = 10
gamma_value = 1.0
stimulus_value = [10 * gamma_value, 10 * gamma_value, 10 * gamma_value]
model_value = 2  # 1 = First auxiliary process (X^(1)), 2 = Second auxiliary process (X^(2))

#%% Reading Samples and variables [Paper results]


# probability_values = np.genfromtxt("../../Samples/Matrix-established-{}.csv".format(1), delimiter=",")
probability_values = np.genfromtxt("../../Matrix-{}.csv".format(1), delimiter=",")
dimension_value = probability_values.shape[0]

if 1 < 3:
    if new_clone_is_soft:
        nu_value = np.genfromtxt("../../Nu-Matrix-Soft.csv", delimiter=",")
    else:
        nu_value = np.genfromtxt("../../Nu-Matrix-Hard.csv", delimiter=",")
else:
    if new_clone_is_soft:
        nu_value = np.genfromtxt("../../Nu-Matrix-Soft-(D).csv", delimiter=",")
    else:
        nu_value = np.genfromtxt("../../Nu-Matrix-Hard-(D).csv", delimiter=",")
nu_value = nu_value * n_mean_value

#%% Reading samples and variables [LHS]


# file = open('Samples.bin', 'rb')
# load_data = pickle.load(file)
#
# dimension_value = load_data[0]
# strata_number = load_data[1]
# samples = load_data[2]
#
# del load_data
# file.close()

#%% Creating Probability matrix [LHS]


# sample_values = list(samples[1])
# probability_values = probability_matrix(dimension_value, stimulus_value, sample_values)

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
    state_list : List[int]
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
    

def main_diagonal_matrices(level, max_level, dimension, probability, mu, nu, stimulus, model):
    """
    Creates the diagonal matrix A_{level, level}.

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
    nu : float
        Niche overlap.
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
    matrix_shape = (int(comb(level - 1, dimension - 1)), int(comb(level - 1, dimension - 1)))
    
    states = level_states(level, dimension)
    
    if level < max_level:
        for state in states:
            data.append(-delta(state, probability, mu, dimension, nu, stimulus, model))
            pos.append(level_position(level, dimension, state))
    else:
        for state in states:
            data.append(-death_delta(state, mu, model))
            pos.append(level_position(level, dimension, state))
        
    md_matrix = coo_matrix((data, (pos, pos)), matrix_shape).tocsc()
    
    return md_matrix


def death_diagonal_matrices(level, dimension, mu, model):
    """
    Creates the sub-diagonal matrix A_{level, level - 1}.

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
    
    matrix_shape = (int(comb(level - 1, dimension - 1)), int(comb(level - 2, dimension - 1)))
    
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


def birth_diagonal_matrices(level, dimension, probability, nu, stimulus):
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
    nu : float
        Niche overlap.
    stimulus : List[float]
        Stimulus parameters.

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
            
            data.append(birth_rate(state, probability, i, dimension, nu, stimulus))
            cols.append(level_position(level + 1, dimension, new_state))
            rows.append(level_position(level, dimension, state))
    
    bd_matrix = coo_matrix((data, (rows, cols)), matrix_shape).tocsc()
    
    return bd_matrix

#%% Linear level reduction algorithm


matrices = [[], [], []]

# Calculating main diagonal matrices
for level_value in range(dimension_value, max_level_value + 1):
    matrix = main_diagonal_matrices(level_value, max_level_value, dimension_value, probability_values, mu_value, nu_value, stimulus_value, model_value)
    matrices[0].append(matrix)

# Calculating lower diagonal matrices
for level_value in range(dimension_value + 1, max_level_value + 1):
    matrix = death_diagonal_matrices(level_value, dimension_value, mu_value, model_value)
    matrices[1].append(matrix)

# Calculating upper diagonal matrices
for level_value in range(dimension_value, max_level_value):
    matrix = birth_diagonal_matrices(level_value, dimension_value, probability_values, nu_value, stimulus_value)
    matrices[2].append(matrix)

# Calculating the inverses of H matrices, and storing them in inverse order
h_matrices = [inv(matrices[0][-1])]

for level_value in range(len(matrices[0]) - 1):
    gc.collect()
    matrix = matrices[0][-(level_value + 2)]
    matrix_term = matrices[2][-(level_value + 1)].dot(h_matrices[-1].dot(matrices[1][-(level_value + 1)]))
    matrix -= matrix_term
    matrix = np.linalg.inv(matrix.todense())
    h_matrices.append(csc_matrix(matrix))


# Calculating the relative values of the distribution
distribution = [np.array([1])]

for level_value in range(len(h_matrices) - 1):
    value = (distribution[level_value] * (-1)) * matrices[2][level_value].dot(h_matrices[-(level_value + 2)])
    distribution.append(value.flatten())

# Normalising the values of the distribution
subTotals = [level.sum() for level in distribution]
total = sum(subTotals)

for level_value in range(len(distribution)):
    distribution[level_value] = distribution[level_value] / total

#%% Storing results


file = open('Parameters.bin', 'wb')
parameters = (["dimension_value", "max_level_value", "mu_value", "gamma_value", "stimulus_value", "model_value"], dimension_value, max_level_value, mu_value, gamma_value, stimulus_value, model_value)
pickle.dump(parameters, file)
file.close()

file = open('Data-{0}-{1}.bin'.format(model_value, 1), 'wb')
pickle.dump(distribution, file)
file.close()
