#%% Packages

from scipy.special import comb
from itertools import chain, combinations
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import inv
import numpy as np
import random,math,pickle

#%% Functions

def cloneSets(dimension,clone):
    """
    Creates an ordered list of tuples representing all subsets of a set of n elements that include the c-th element.

    Parameters
    ----------
    dimension : int
        Number of elements.
    clone : int
        Speficied element (starts at 0).

    Returns
    -------
    list
        list of tuples representing all subsets of a set of dimension elements that include the clone-th element.

    """
    
    if clone >= dimension or clone < 0:
        return -1
    
    x = range(dimension)
    sets = list(chain(*[combinations(x,ni) for ni in range(dimension + 1)]))
    d = []
    
    for T in sets:
        if (clone not in T):
            d.insert(0,sets.index(T))
    
    for i in d:
        sets.pop(i)
        
    return sets

def isAbsorbed(state):
    """
    Checks if a state has been absorbed.

    Parameters
    ----------
    state : list
        List of number of cells per clonotype.

    Returns
    -------
    bool
        True if any component of state is 0, False otherwise.

    """

    
    for i in range(len(state)):
        if state[i]<=0:
            return True
    
    return False

def isUnit(state):
    """
    Checks if a state consists of only one cell per clonotype.

    Parameters
    ----------
    state : list
        List of number of cells per clonotype.

    Returns
    -------
    bool
        True if every component of state is 1, False otherwise.

    """
    
    for i in range(len(state)):
        if state[i] != 1:
            return False
        
    return True

def position(level,dimension,state):
    """
    Calculates the position of state in level.

    Parameters
    ----------
    level : int
        Level of the state space.
    dimension : int
        Number of clonotypes.
    state : list
        List of number of cells per clonotype.

    Returns
    -------
    int
        Position of state in level, or -1 if state is not in the elvel.

    """
    
    if len(state)!=dimension or sum(state)!=level or isAbsorbed(state):
        return -1
    
    if level == dimension and isUnit(state):
        return 0
    
    position = 0
    
    maxCells = level - dimension + 1

    for i in range(dimension):
        position += (state[i] - 1) * (maxCells ** i)
        
    for i in range(dimension - 2):
        position += (state[dimension - 1 - i] - 1) * (1 - (maxCells ** (dimension - 1 - i)))
    
    position = int(position / (maxCells - 1))
    
    for i in range(dimension - 2):
        position += int(comb(level - 1 - sum(state[dimension - i:dimension]), dimension - 1 - i)) - int(comb(level - sum(state[dimension -i - 1:dimension]), dimension - 1 - i))   
    
    return int(position - 1)

def isInLevel(state,level,dimension):
    """
    Determines if a state is part of a level.

    Parameters
    ----------
    state : list
        List of number of cells per clonotype.
    level : int
        Level of the state space.
    dimension : int
        Number of clonotypes.

    Returns
    -------
    bool
        True if state is part of level, False otherwise..

    """
    
    if len(state)!=dimension or sum(state)!=level or isAbsorbed(state):
        return False
    else:
        return True
    

def levelStates(level,dimension):
    """
    Creates a list of all states in level.

    Parameters
    ----------
    level : int
        Level of the state space.
    dimension : int
        Number of clonotypes.

    Returns
    -------
    levelStates : list
        List of all states in level.

    """

    levelStates = []
    n = [1 for _ in range(dimension)]

    while True:

        if isInLevel(n, level, dimension):
            levelStates.append(n[:])

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

    return levelStates

def probabilityRow(row,probability,matrix,location,stimulus):
    """
    Calculates the values of a row of the probability matrix.

    Parameters
    ----------
    row : int
        Row of the matrix to be updated.
    probability : list
        Specified probability value.
    matrix : list
        Probability matrix.
    location : int
        Position of the first selected probability.
    stimulus : list
        Stimulus parameters.

    Returns
    -------
    probabilities : list
        List of values for row.

    """
    
    probabilities = [0 for _ in range(len(matrix[0]))]
    size = len(probabilities)
    
    if row == 0:
        probabilities[location] = probability
        for i in range(size):
            if i != location:
                probabilities[i] = float((1 - probability) / (size - 1))
    else:
        free = []
        i = size - 1
        while i >= 0:
            pair = matrix[row][i]
            if pair != [row,i]:
                probabilities[i] = (stimulus[pair[0]] / stimulus[row]) * matrix[pair[0]][pair[1]]
            else:
                free.append(i)
            i -= 1
        
        if len(free) > 0:
            value = (1 - sum(probabilities)) / len(free)
            for i in free:
                probabilities[i] = value
    
    return probabilities

def probabilityMatrix(probability,dimension,location,stimulus):
    """
    Creates the probability matrix for a probability and location.

    Parameters
    ----------
    probability : float
        Probability value.
    dimension : int
        Number of clonotypes.
    location : int
        Position of the first selected probability.
    stimulus : list
        Stimulus parameters.

    Returns
    -------
    matrix : list
        list expression of the probability matrix.

    """
    
    sets = []
    matrix = [[0 for _ in range(2 ** (dimension - 1))] for _ in range(dimension)]
    
    for i in range(dimension):
        sets.append(cloneSets(dimension,i))
        
    for row in range(len(matrix)):
        for col in range(len(matrix[0])):
            for i in range(len(matrix)):
                for j in range(len(matrix[0])):
                    if matrix[row][col] == 0 and sets[row][col] == sets[i][j]:
                        matrix[row][col] = [i,j]
                        break
                else:
                    continue
                break
            
    for i in range(len(matrix)):
        matrix[i] = probabilityRow(i,probability,matrix,location,stimulus)
                        
    return matrix

def sumClones(subset,state):
    """
    Sums the number of cells in a subset of clonotypes for a state.

    Parameters
    ----------
    subset : tuple
        Clonotypes in the subset.
    state : list
        Number of cells per clonotype.

    Returns
    -------
    float
        Total number of cells in subset for state.

    """
    
    total = 0.0
    
    for s in subset:
        total += float(state[s])
    
    return float(total)


def birthRate(state,probability,clone,dimension,stimulus):
    """
    Calculates the birth rate for clone in state.

    Parameters
    ----------
    state : list
        Number of cells per clonotype.
    probability : list
        Probability matrix.
    clone : int
        Specified clone.
    dimension : int
        Number of clonotypes.
    stimulus : list
        Stimulus parameters.

    Returns
    -------
    float
        Birth rate for clone in state.

    """
    
    rate = 0.0
    sets = cloneSets(dimension ,clone)
    
    for i in range(len(sets)):
        if sumClones(sets[i],state) != 0:
            rate += probability[clone][i] / sumClones(sets[i],state)
    
    return rate * state[clone] * stimulus[clone]

def deathRate(state,clone,mu,model):
    """
    calculates the death rate for clone in state for a given model type.

    Parameters
    ----------
    state : list
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
    
    if model == 0:
        if state[clone] > 1:
            return state[clone] * mu
        else:
            return 0.0
    if model == 1:
        return (state[clone] - 1) * mu


def delta(state,probability,mu,dimension,stimulus,model):
    """
    Calculates the sum of all birth and death rates for state.

    Parameters
    ----------
    state : list
        Number of cells per clonotype.
    probability : list
        Probability matrix.
    mu : float
        Single cell death rate.
    dimension : int
        Number of clonotypes.
    stimulus : list
        Stimulus parameters.
    model : int
        Auxiliary process (0 = X^1, 1 = X^2).

    Returns
    -------
    total : float
        Sum of all birth and death rates for state.

    """
    
    total = 0.0
    
    for i in range(len(state)):
        total += deathRate(state, i, mu, model)
            
    for i in range(len(state)):
        total += birthRate(state,probability,i,dimension,stimulus)
        
    return total

def deathDelta(state,mu,model):
    """
    Calculates the sum of all death rates for a state

    Parameters
    ----------
    state : list
        Number of cells per clonotype.
    mu : float
        Single cell death rate.
    model : int
        Auxiliary process (0 = X^1, 1 = X^2).

    Returns
    -------
    total : float
        Sum of all death rates for state.

    """
    
    total = 0.0
    
    for i in range(len(state)):
        total += deathRate(state, i, mu, model)
        
    return total

def mainDiagonalLists(level,maxLevel,dimension,probability,mu,stimulus,model):
    """
    Creates the diagonal matrix A_{level,level}

    Parameters
    ----------
    level : int
        Level in the state space.
    maxLevel : int
        Maximum level of the state space.
    dimension : int
        Number of clonotypes.
    probability : list
        Probability matrix.
    mu : float
        Single cell death rate.
    stimulus : list
        Stimulus parameters.
    model : int
        Auxiliary process (0 = X^1, 1 = X^2).

    Returns
    -------
    matrix : csc_matrix
        Matrix A_{level,level}.

    """
    
    pos = []
    data = []
    
    states = levelStates(level, dimension)
    
    if level < maxLevel:
        for state in states:
            data.append(-delta(state, probability, mu, dimension, stimulus, model))
            pos.append(position(level, dimension, state))
    else:
        for state in states:
            data.append(-deathDelta(state, mu, model))
            pos.append(position(level, dimension, state))
        
    # matrixShape = (int(comb(level - 1,dimension - 1)),int(comb(level - 1,dimension - 1)))
    # matrix = coo_matrix((data,(pos,pos)),matrixShape).tocsc()
    
    return pos,data

def deathDiagonalLists(level,dimension,mu,model):
    """
    Creates the sub-diagonal matrix A_{level,level + 1}

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
    matrix : csc_matrix
        Matrix A_{level,level + 1}.

    """
    
    rows = []
    cols = []
    data = []
    
    states = levelStates(level, dimension)
    
    for state in states:
        for i in range(len(state)):
            newState = state[:]
            newState[i] -= 1
            if isInLevel(newState, level - 1, dimension):
                data.append(deathRate(state, i, mu, model))
                cols.append(position(level - 1, dimension, newState))
                rows.append(position(level, dimension, state))
    
    # matrixShape = (int(comb(level - 1,dimension - 1)),int(comb(level - 2,dimension - 1)))
    # matrix = coo_matrix((data,(rows,cols)),matrixShape).tocsc()
    
    return rows,cols,data

def birthDiagonalLists(level,dimension,probability,stimulus):
    """
    Creates the diagonal matrix A_{level - 1,level}

    Parameters
    ----------
    level : int
        Level in the state space.
    dimension : int
        Number of clonotypes.
    probability : list
        Probability matrix.
    stimulus : list
        Stimulus parameters.

    Returns
    -------
    matrix : csc_matrix
        Matrix A_{level - 1,level}.

    """
    
    rows = []
    cols = []
    data = []
    
    states = levelStates(level, dimension)
    
    for state in states:
        for i in range(len(state)):
            newState = state[:]
            newState[i] += 1
            if isInLevel(newState, level + 1, dimension):
                data.append(birthRate(state, probability, i, dimension, stimulus))
                cols.append(position(level + 1, dimension, newState))
                rows.append(position(level, dimension, state))
    
    # matrixShape = (int(comb(level - 1,dimension - 1)),int(comb(level,dimension - 1)))
    # matrix = coo_matrix((data,(rows,cols)),matrixShape).tocsc()
    
    return rows,cols,data

# def probabilityRow(row,probability,matrix):
#     """Function that updates a row of the probability matrix.

#     Arguments:
#         row - Row of the matrix to be updated.
#         probability - Specified probability value.
#         matrix - Probability matrix.
#     """

#     probabilities = [0 for _ in xrange(len(matrix[0]))]
#     size = len(probabilities)

#     if row == 0:
#         probabilities[location] = probability
#         for i in xrange(size):
#             if i != location:
#                 probabilities[i] = float((1 - probability) / (size - 1))
#     else:
#         free = []
#         i = size - 1
#         while i >= 0:
#             pair = matrix[row][i]
#             if pair != [row,i]:
#                 probabilities[i] = (phi[pair[0]] / phi[row]) * matrix[pair[0]][pair[1]]
#             else:
#                 free.append(i)
#             i -= 1

#         if len(free) > 0:
#             value = (1 - sum(probabilities)) / len(free)
#             for i in free:
#                 probabilities[i] = value

#     return probabilities

# def probabilityMatrix(probability):
#     """Function that creates the probability matrix when setting a specified probability.

#     Arguments:
#         probability - Specified probability value.
#     """

#     sets = []
#     matrix = [[0 for _ in xrange(2 ** (N - 1))] for _ in xrange(N)]

#     for i in xrange(N):
#         sets.append(cloneSets(N,i))

#     for row in xrange(len(matrix)):
#         for col in xrange(len(matrix[0])):
#             for i in xrange(len(matrix)):
#                 for j in xrange(len(matrix[0])):
#                     if matrix[row][col] == 0 and sets[row][col] == sets[i][j]:
#                         matrix[row][col] = [i,j]
#                         break
#                 else:
#                     continue
#                 break

#     for i in xrange(len(matrix)):
#         matrix[i] = probabilityRow(i,probability,matrix)

#     return matrix

# def position(state):
#     """Function to obtain the position in the coefficient matrix.

#     Returns the position in the coefficient matrix corresponding to the state specified, if the state is not in the state space returns -1.

#     Arguments:
#         state - State list (number of cells per clonotype).
#     """

#     for i in xrange(N):
#         if state[i] > eta or state[i] < 0:
#             return -1

#     place = 0
#     for i in xrange(len(state)):
#         place += state[i] * ((eta + 1) ** i)

#     return place


# def sumClones(subset,state):
#     """Function that sums the number of cells in the specified subset of clonotypes.

#     Arguments:
#         subset - tuple of the clonotypes in the subset.
#         state - State list (number of cells per clonotype).
#     """

#     total = 0.0

#     for s in subset:
#         total += float(state[s])

#     return float(total)


# def birthRate(state,probability,clone):
#     """Function that calculates the birth rate for a given state, clone and probability vector.

#     Arguments:
#         state - State list (number of cells per clonotype).
#         probability - Probability list.
#         clone - Specified clone.
#     """

#     rate = 0.0
#     sets = cloneSets(N,clone)

#     for i in xrange(len(sets)):
#         if sumClones(sets[i],state) != 0:
#             rate += probability[clone][i] / sumClones(sets[i],state)

#     return rate * state[clone] * phi[clone]


# def delta(state,probability):
#     """Function to calculate the sum of all death and birth rates for a given state.

#     Arguments:
#         state - State list (number of cells per clonotype).
#         probability - Probability list.
#     """

#     total = 0.0

#     for i in xrange(len(state)):
#         if state[i] > 0:
#             total += state[i] * mu
#             total += birthRate(state,probability,i)

#     return total

# def validStates(state):
#     """Function that creates a dictionary of states (casted as tuples and used as keys) that can reach the desired state and their new relative position (used as values).

#     Arguments:
#         state - State to be visited
#     """

#     ValidStates = dict()
#     StatePosition = 0
#     n = [0 for _ in xrange(N)]

#     while True:
#         escape = False
#         absorbed = False

#         for i in xrange(N):
#             if n[i] == 0 and state[i]>0:
#                 escape = True
#                 break

#         if n != state:
#             for i in xrange(N):
#                 if n[i] == 0:
#                     absorbed = True

#         if not escape and not absorbed:
#             ValidStates[tuple(n)] = StatePosition
#             StatePosition += 1

#         n[0] += 1
#         for i in xrange(len(n)):
#             if n[i] > eta:
#                 if (i + 1) < len(n):
#                     n[i+1] += 1
#                     n[i] = 0
#                 for j in xrange(i):
#                     n[j] = 0

#         if n[-1] > eta:
#             break

#     return ValidStates

# def coefficientMatrix(probability,dimensions,state,initialStates):
#     """Function that creates the coefficient matrix of the difference equations as a csr_matrix.

#     Arguments:
#         probability - Probability matrix.
#         dimensions - Dimensions of the matrix.
#         state - State to be visited.
#         initialStates - Initial states with non-zero probability of visiting the specified state.
#     """

#     rows = []
#     cols = []
#     data = []

#     for key in initialStates:
#         n = list(key)
#         current_row = initialStates[key]

#         if n == state:
#             starting_visit = True
#             data.append(1)
#         else:
#             starting_visit = False
#             data.append(-delta(n,probability))
#             current_delta = len(data) - 1

#         rows.append(current_row)
#         cols.append(current_row)

#         if not starting_visit:
#             for l in xrange(N):
#                 temp = n[:]
#                 temp[l] += 1
#                 if tuple(temp) in initialStates:
#                     current_col  = initialStates[tuple(temp)]
#                     rows.append(current_row)
#                     cols.append(current_col)
#                     data.append(birthRate(n,probability,l))
#                 elif temp[l] == eta + 1:
#                     data[current_delta] += birthRate(n,probability,l)

#                 temp[l] -= 2
#                 if tuple(temp) in initialStates:
#                     current_col = initialStates[tuple(temp)]
#                     rows.append(current_row)
#                     cols.append(current_col)
#                     data.append(n[l] * mu)

#     matrix = coo_matrix((data,(rows,cols)),dimensions).tocsr()

#     return matrix

# def nonHomogeneousTerm(size,state,initialStates):
#     """Function that creates the vector of non-homogenous terms for the system of difference equations.
    
#     Arguments:
#         size - Length of the vector.
#         state - State to be visited.
#         initialStates - Initial states with non-zero probability of visiting the specified state.
#     """

#     b = [0] * size
#     b[initialStates[tuple(state)]] = 1

#     return b

#%% Solving the matrix equation

dimension = 3
maxLevel = 99
location = (2 ** (dimension - 1)) - 1
mu = 1.0
gamma = 1.0
stimulus = [5 * gamma,10 * gamma,10 * gamma]
dp = 0.05

p = 0.0
Solutions = []
prob = []

while p < 1+dp:

    P = probabilityMatrix(p)
    M = coefficientMatrix(P,(len(initialStates),len(initialStates)),v_state,initialStates)
    b = nonHomogeneousTerm(len(initialStates),v_state,initialStates)

    X_initial = spsolve(M,b)
    X = [0] * ((eta + 1) ** N)
    for key in initialStates:
        n = list(key)
        X[position(n)] = X_initial[initialStates[key]]

    Solutions.append(X)
    prob.append(p)

    #Writing to console
    with open('console.txt','a') as file:
        file.write('Run {0} CPU time: {1:.3f}s\n'.format(run,time.clock() - run_start))

    p += dp
    run += 1

#%% Storing Data

with open('Data.bin','wb') as file:
    data = (Solutions,prob,N,eta,mu,phi,gamma)
    pickle.dump(data,file)

# file =  open('Parameters.bin','wb')
# data = (dimension,maxLevel,location,firstProbability,mu,gamma,stimulus,model)
# pickle.dump(data,file)
# file.close()
    
# file = open('Data.bin','wb')
# pickle.dump(values, file)
# file.close()