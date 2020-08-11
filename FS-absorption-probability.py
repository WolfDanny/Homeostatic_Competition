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
    position - 1 : int
        Position of state in level, or -1 if state is not in the level.

    """
    
    if level == dimension and isUnit(state):
        return 0
    
    if len(state)!=dimension or sum(state)!=level or (state.count(0) > 0):
        return -1
    
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

def totalPosition(dimension,state):
    """
    Calculates the position of state in the non-absorbed state space.

    Parameters
    ----------
    dimension : int
        Number of clonotypes.
    state : list
        List of number of cells per clonotype.

    Returns
    -------
    Position : int
        Position of state in the non-absorbed state space.
    """
    
    level = sum(state)
    Position = 0
    
    if level > dimension:
        for i in range(1, level - dimension + 1):
            Position += comb(level - i - 1, dimension - 1)
            
    Position += position(level, dimension, state)
            
    return int(Position)

def absorbedPosition(dimension,state,maxLevel):
    """
    Calculates the position of state in the absorbed state space.

    Parameters
    ----------
    dimension : int
        Number of clonotypes.
    state : list
        List of number of cells per clonotype.
    maxLevel : int
        Maximum level of the state space.

    Returns
    -------
    Position : int
        Position of state in the absorbed state space.

    """
    
    if not (state.count(0) > 0):
        return -1
    elif state.count(0) > 1:
        return -1
    elif sum(state) > maxLevel:
        return -1
    elif len(state) != dimension:
        return -1
    
    projection = state[:]
    projectionComponent = projection.index(0)
    projection.pop(projectionComponent)
    
    Position = projectionComponent * comb(maxLevel,dimension - 1)
    Position += totalPosition(dimension - 1, projection)
    
    return int(Position)

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
    
    if len(state)!=dimension or sum(state)!=level or (state.count(0) > 0):
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
    stateList : list
        List of all states in level.

    """

    stateList = []
    n = [1 for _ in range(dimension)]

    while True:

        if isInLevel(n, level, dimension):
            stateList.append(n[:])

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

    return stateList

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
    Calculates the death rate for clone in state for a given model type.

    Parameters
    ----------
    state : list
        Number of cells per clonotype.
    clone : int
        Specified clone.
    mu : float
        Single cell death rate.
    model : int
        Process (0 = X, 1 = X^1, 2 = X^2).

    Returns
    -------
    float
        Death rate for clone in state.

    """
    if model == 0:
        return state[clone] * mu
    if model == 1:
        if state[clone] > 1:
            return state[clone] * mu
        else:
            return 0.0
    if model == 2:
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
        Process (0 = X, 1 = X^1, 2 = X^2).

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
        Process (0 = X, 1 = X^1, 2 = X^2).

    Returns
    -------
    total : float
        Sum of all death rates for state.

    """
    
    total = 0.0
    
    for i in range(len(state)):
        total += deathRate(state, i, mu, model)
        
    return total

def transitionMatrix(maxLevel,dimension,mu,probability,stimulus,model):
    """
    Creates the transition matrix for the embedded Markov process as a csc_matrix

    Parameters
    ----------
    maxLevel : int
        Maximum level of the state space.
    dimension : int
        Number of clonotypes.
    mu : float
        Single cell death rate.
    probability : list
        Probability matrix.
    stimulus : list
        Stimulus parameters.
    model : int
        Process (0 = X, 1 = X^1, 2 = X^2).

    Returns
    -------
    matrix : csc_matrix
        Transition matrix of the embedded Markov process.

    """
    
    values = []
    rows = []
    cols = []
    
    deathRows, deathCols, deathVals = deathDiagonalLists(maxLevel, dimension, mu, probability, stimulus, model)
    rows.append(deathRows)
    cols.append(deathCols)
    values.append(deathVals)
    
    birthRows, birthCols, birthVals = birthDiagonalLists(maxLevel, dimension, mu, probability, stimulus, model)
    rows.append(birthRows)
    cols.append(birthCols)
    values.append(birthVals)
    
    matrixShape = (comb(maxLevel, dimension), comb(maxLevel, dimension))
    matrix = coo_matrix((values,(rows,cols)),matrixShape).tocsc()
    
    return matrix
    

def deathDiagonalLists(maxLevel,dimension,mu,probability,stimulus,model):
    """
    Creates the lists required to create the sub-diagonal matrices A_{level,level + 1}

    Parameters
    ----------
    maxLevel : int
        Maximum level of the state space.
    dimension : int
        Number of clonotypes.
    mu : float
        Single cell death rate.
    probability : list
        Probability matrix.
    stimulus : list
        Stimulus parameters.
    model : int
        Process (0 = X, 1 = X^1, 2 = X^2).

    Returns
    -------
    rows : list
        List of the row positions of the value in the same location in 'values'.
    cols : list
        List of the column positions of the value in the same location in 'values'.
    values : list
        List of the entries of the transition matrix.

    """
    
    rows = []
    cols = []
    values = []
    
    for level in range(dimension + 1,maxLevel + 1):
    
        states = levelStates(level, dimension)
        
        if level < maxLevel:
            for state in states:
                for i in range(len(state)):
                    newState = state[:]
                    newState[i] -= 1
                    if isInLevel(newState, level - 1, dimension):
                        values.append(deathRate(state, i, mu, model) / delta(state, probability, mu, dimension, stimulus, model))
                        cols.append(totalPosition(dimension, newState))
                        rows.append(totalPosition(dimension, state))
        else:
            for state in states:
                for i in range(len(state)):
                    newState = state[:]
                    newState[i] -= 1
                    if isInLevel(newState, level - 1, dimension):
                        values.append(deathRate(state, i, mu, model) / deathDelta(state, mu, model))
                        cols.append(totalPosition(dimension, newState))
                        rows.append(totalPosition(dimension, state))
    
    return rows,cols,values

def birthDiagonalLists(maxLevel,dimension,mu,probability,stimulus,model):
    """
    Creates the lists required to create the super-diagonal matrices A_{level - 1,level}

    Parameters
    ----------
    maxLevel : int
        Maximum level of the state space.
    dimension : int
        Number of clonotypes.
    mu : float
        Single cell death rate.
    probability : list
        Probability matrix.
    stimulus : list
        Stimulus parameters.
    model : int
        Process (0 = X, 1 = X^1, 2 = X^2).

    Returns
    -------
    rows : list
        List of the row positions of the value in the same location in 'values'.
    cols : list
        List of the column positions of the value in the same location in 'values'.
    values : list
        List of the entries of the transition matrix.

    """
    
    rows = []
    cols = []
    values = []
    
    for level in range(dimension,maxLevel):
    
        states = levelStates(level, dimension)
    
        for state in states:
            for i in range(len(state)):
                newState = state[:]
                newState[i] += 1
                if isInLevel(newState, level + 1, dimension):
                    values.append(birthRate(state, probability, i, dimension, stimulus) / delta(state, probability, mu, dimension, stimulus, model))
                    cols.append(totalPosition(level + 1, dimension, newState))
                    rows.append(totalPosition(level, dimension, state))

    
    return rows,cols,values

def absorptionMatrix(maxLevel,dimension,mu,probability,stimulus,model):
    """
    Creates the transition matrix from A^{0} to A^{1} in the embedded Markov chain as a csc_matrix

    Parameters
    ----------
    maxLevel : int
        Maximum level of the state space.
    dimension : int
        Number of clonotypes.
    mu : float
        Single cell death rate.
    probability : list
        Probability matrix.
    stimulus : list
        Stimulus parameters.
    model : int
        Process (0 = X, 1 = X^1, 2 = X^2).

    Returns
    -------
    matrix : csc_matrix
        Transition matrix from A^{0} to A^{1} in the embedded Markov process.

    """
    
    rows = []
    cols = []
    values = []
    
    for level in range(dimension,maxLevel + 1):
        
        states = levelStates(level, dimension)
        
        if level < maxLevel:
            for state in states:
                for i in range(len(state)):
                    newState = state[:]
                    newState[i] -= 1
                    if state.count(0) == 1:
                        values.append(deathRate(state, i, mu, model) / delta(state, probability, mu, dimension, stimulus, model))
                        cols.append(absorbedPosition(dimension, state, maxLevel))
                        rows.append(totalPosition(dimension, state))
        else:
            for state in states:
                for i in range(len(state)):
                    newState = state[:]
                    newState[i] -= 1
                    if state.count(0) == 1:
                        values.append(deathRate(state, i, mu, model) / deathDelta(state, mu, model))
                        cols.append(absorbedPosition(dimension, state, maxLevel))
                        rows.append(totalPosition(dimension, state))
    
    matrixShape = (comb(maxLevel, dimension), dimension * comb(maxLevel, dimension - 1))
    matrix = coo_matrix((values,(rows,cols)),matrixShape).tocsc()
    
    return matrix
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