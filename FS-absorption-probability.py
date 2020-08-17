#%% Packages

from scipy.special import comb
from scipy.stats import uniform
from scipy.sparse import coo_matrix, identity
from scipy.sparse.linalg import spsolve
from itertools import chain, combinations
import numpy as np
import random,math,pickle,functools,operator

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
    
    if level == dimension and state.count(1) == dimension:
        return 0
    
    if len(state)!=dimension or sum(state)!=level or state.count(0) > 0:
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

        if len(n) == dimension and sum(n) == level and (n.count(0) == 0):
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

def probabilityMatrix(dimension,stimulus,sample):
    """
    Creates the probability matrix from a sample of values.

    Parameters
    ----------
    dimension : int
        Number of clonotypes.
    stimulus : list
        Stimulus parameters.
    sample : list
        List of probability values sampled.

    Returns
    -------
    matrix : list
        list expression of the probability matrix.

    """
    
    sampleLocal = sample[:]
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
    
    del sets
    
    rowValues = [0 for _ in range(len(matrix[0]))]
    for pos, pair in reversed(list(enumerate(matrix[0]))):
        if pair != [0,0]:
            rowValues[pos] = uniform(loc=0,scale=1-sum(rowValues)).ppf(sampleLocal.pop())
        else:
            rowValues[pos] = 1 - sum(rowValues)
    matrix[0] = rowValues[:]
            
    for i in range(1,len(matrix)):
        rowValues = [0 for _ in range(len(matrix[i]))]
        for pos, pair in reversed(list(enumerate(matrix[i]))):
            if pair != [i,pos]:
                rowValues[pos] = matrix[pair[0]][pair[1]] * (stimulus[pair[0]] / stimulus[i])
            elif pair != [i,0]:
                rowValues[pos] = uniform(loc=0,scale=1-sum(rowValues)).ppf(sampleLocal.pop())
            elif pair == [i,0]:
                rowValues[pos] = 1 - sum(rowValues)
        matrix[i] = rowValues[:]
                        
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
    
    
    rows = functools.reduce(operator.iconcat, rows, [])
    cols = functools.reduce(operator.iconcat, cols, [])
    values = functools.reduce(operator.iconcat, values, [])
    
    
    matrixShape = (int(comb(maxLevel, dimension)), int(comb(maxLevel, dimension)))
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
                    if newState.count(0) == 0:
                        values.append(deathRate(state, i, mu, model) / delta(state, probability, mu, dimension, stimulus, model))
                        cols.append(totalPosition(dimension, newState))
                        rows.append(totalPosition(dimension, state))
        else:
            for state in states:
                for i in range(len(state)):
                    newState = state[:]
                    newState[i] -= 1
                    if newState.count(0) == 0:
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
                
                values.append(birthRate(state, probability, i, dimension, stimulus) / delta(state, probability, mu, dimension, stimulus, model))
                cols.append(totalPosition(dimension, newState))
                rows.append(totalPosition(dimension, state))
    
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
                    if newState.count(0) == 1:
                        values.append(deathRate(state, i, mu, model) / delta(state, probability, mu, dimension, stimulus, model))
                        cols.append(absorbedPosition(dimension, newState, maxLevel))
                        rows.append(totalPosition(dimension, state))
        else:
            for state in states:
                for i in range(len(state)):
                    newState = state[:]
                    newState[i] -= 1
                    if newState.count(0) == 1:
                        values.append(deathRate(state, i, mu, model) / deathDelta(state, mu, model))
                        cols.append(absorbedPosition(dimension, newState, maxLevel))
                        rows.append(totalPosition(dimension, state))
    
    matrixShape = (int(comb(maxLevel, dimension)), int(dimension * comb(maxLevel, dimension - 1)))
    matrix = coo_matrix((values,(rows,cols)),matrixShape).tocsc()
    
    return matrix

#%% Reading samples and variables

file = open('Samples.bin','rb')
load_data = pickle.load(file)

dimension = load_data[0]
strata = load_data[1]
samples = load_data[2]

del load_data
file.close()

#%% Variables

maxLevel = 179 # To have ~10^6 states in the HPC set to 179
mu = 1.0
gamma = 1.0
stimulus = [5 * gamma,10 * gamma,10 * gamma]
model = 0 # 0 = X, 1 = X^1, 2 = X^2

#%% Solving the matrix equation

sampleNumber = Placeholder
sample = samples[sampleNumber]

shape = int(comb(maxLevel, dimension))

values = list(sample)

ProbMatrix = probabilityMatrix(dimension, stimulus, values)
TransMatrix = transitionMatrix(maxLevel, dimension, mu, ProbMatrix, stimulus, model)
AbsMatrix = absorptionMatrix(maxLevel, dimension, mu, ProbMatrix, stimulus, model)
AssocMatrix = identity(shape,format="csc") - TransMatrix

AbsorptionDistribution = spsolve(AssocMatrix, AbsMatrix)

#%% Storing Data

file = open('Parameters.bin','wb')
data = (["dimension","maxLevel","mu","gamma","stimulus","strata","model"],dimension,maxLevel,mu,gamma,stimulus,strata,model)
pickle.dump(data,file)
file.close()
    
file = open('Data-'+str(sampleNumber)+'.bin','wb')
pickle.dump(AbsorptionDistribution, file)
file.close()
