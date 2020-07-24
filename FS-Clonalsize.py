import numpy as np
import pickle,time
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve
from itertools import chain, combinations

N = 3 #Number of clonotypes
eta = 45 #Maximum number of cells per clonotype
location = (2 ** (N - 1)) - 1 #Probability to be ploted
v_state = [10,10,10] #State to be visited
mu = 1.0 #Death rate
gamma = 1.0 #Stimulus rate
phi = [5 * gamma,10 * gamma,10 * gamma] #Vector of stimulus parameters
dp = 0.05

#%% Functions

def cloneSets(number,clone):
    """Function that creates an ordered list of tuples representing all subsets of a set of n elements that include the c-th element.

    Returns -1 if c is not in [0,c).

    Arguments:
        number - Number of elements.
        clone - Speficied element (starts at 0).
    """

    if clone >= number or clone < 0:
        return -1

    x = xrange(number)
    sets = list(chain(*[combinations(x,ni) for ni in xrange(number + 1)]))
    d = []

    for T in sets:
        if (clone not in T):
            d.insert(0,sets.index(T))

    for i in d:
        sets.pop(i)

    return sets

def probabilityRow(row,probability,matrix):
    """Function that updates a row of the probability matrix.

    Arguments:
        row - Row of the matrix to be updated.
        probability - Specified probability value.
        matrix - Probability matrix.
    """

    probabilities = [0 for _ in xrange(len(matrix[0]))]
    size = len(probabilities)

    if row == 0:
        probabilities[location] = probability
        for i in xrange(size):
            if i != location:
                probabilities[i] = float((1 - probability) / (size - 1))
    else:
        free = []
        i = size - 1
        while i >= 0:
            pair = matrix[row][i]
            if pair != [row,i]:
                probabilities[i] = (phi[pair[0]] / phi[row]) * matrix[pair[0]][pair[1]]
            else:
                free.append(i)
            i -= 1

        if len(free) > 0:
            value = (1 - sum(probabilities)) / len(free)
            for i in free:
                probabilities[i] = value

    return probabilities

def probabilityMatrix(probability):
    """Function that creates the probability matrix when setting a specified probability.

    Arguments:
        probability - Specified probability value.
    """

    sets = []
    matrix = [[0 for _ in xrange(2 ** (N - 1))] for _ in xrange(N)]

    for i in xrange(N):
        sets.append(cloneSets(N,i))

    for row in xrange(len(matrix)):
        for col in xrange(len(matrix[0])):
            for i in xrange(len(matrix)):
                for j in xrange(len(matrix[0])):
                    if matrix[row][col] == 0 and sets[row][col] == sets[i][j]:
                        matrix[row][col] = [i,j]
                        break
                else:
                    continue
                break

    for i in xrange(len(matrix)):
        matrix[i] = probabilityRow(i,probability,matrix)

    return matrix

def position(state):
    """Function to obtain the position in the coefficient matrix.

    Returns the position in the coefficient matrix corresponding to the state specified, if the state is not in the state space returns -1.

    Arguments:
        state - State list (number of cells per clonotype).
    """

    for i in xrange(N):
        if state[i] > eta or state[i] < 0:
            return -1

    place = 0
    for i in xrange(len(state)):
        place += state[i] * ((eta + 1) ** i)

    return place


def sumClones(subset,state):
    """Function that sums the number of cells in the specified subset of clonotypes.

    Arguments:
        subset - tuple of the clonotypes in the subset.
        state - State list (number of cells per clonotype).
    """

    total = 0.0

    for s in subset:
        total += float(state[s])

    return float(total)


def birthRate(state,probability,clone):
    """Function that calculates the birth rate for a given state, clone and probability vector.

    Arguments:
        state - State list (number of cells per clonotype).
        probability - Probability list.
        clone - Specified clone.
    """

    rate = 0.0
    sets = cloneSets(N,clone)

    for i in xrange(len(sets)):
        if sumClones(sets[i],state) != 0:
            rate += probability[clone][i] / sumClones(sets[i],state)

    return rate * state[clone] * phi[clone]


def delta(state,probability):
    """Function to calculate the sum of all death and birth rates for a given state.

    Arguments:
        state - State list (number of cells per clonotype).
        probability - Probability list.
    """

    total = 0.0

    for i in xrange(len(state)):
        if state[i] > 0:
            total += state[i] * mu
            total += birthRate(state,probability,i)

    return total

def validStates(state):
    """Function that creates a dictionary of states (casted as tuples and used as keys) that can reach the desired state and their new relative position (used as values).

    Arguments:
        state - State to be visited
    """

    ValidStates = dict()
    StatePosition = 0
    n = [0 for _ in xrange(N)]

    while True:
        escape = False
        absorbed = False

        for i in xrange(N):
            if n[i] == 0 and state[i]>0:
                escape = True
                break

        if n != state:
            for i in xrange(N):
                if n[i] == 0:
                    absorbed = True

        if not escape and not absorbed:
            ValidStates[tuple(n)] = StatePosition
            StatePosition += 1

        n[0] += 1
        for i in xrange(len(n)):
            if n[i] > eta:
                if (i + 1) < len(n):
                    n[i+1] += 1
                    n[i] = 0
                for j in xrange(i):
                    n[j] = 0

        if n[-1] > eta:
            break

    return ValidStates

def coefficientMatrix(probability,dimensions,state,initialStates):
    """Function that creates the coefficient matrix of the difference equations as a csr_matrix.

    Arguments:
        probability - Probability matrix.
        dimensions - Dimensions of the matrix.
        state - State to be visited.
        initialStates - Initial states with non-zero probability of visiting the specified state.
    """

    rows = []
    cols = []
    data = []

    for key in initialStates:
        n = list(key)
        current_row = initialStates[key]

        if n == state:
            starting_visit = True
            data.append(1)
        else:
            starting_visit = False
            data.append(-delta(n,probability))
            current_delta = len(data) - 1

        rows.append(current_row)
        cols.append(current_row)

        if not starting_visit:
            for l in xrange(N):
                temp = n[:]
                temp[l] += 1
                if tuple(temp) in initialStates:
                    current_col  = initialStates[tuple(temp)]
                    rows.append(current_row)
                    cols.append(current_col)
                    data.append(birthRate(n,probability,l))
                elif temp[l] == eta + 1:
                    data[current_delta] += birthRate(n,probability,l)

                temp[l] -= 2
                if tuple(temp) in initialStates:
                    current_col = initialStates[tuple(temp)]
                    rows.append(current_row)
                    cols.append(current_col)
                    data.append(n[l] * mu)

    matrix = coo_matrix((data,(rows,cols)),dimensions).tocsr()

    return matrix

def nonHomogeneousTerm(size,state,initialStates):
    """Function that creates the vector of non-homogenous terms for the system of difference equations.
    
    Arguments:
        size - Length of the vector.
        state - State to be visited.
        initialStates - Initial states with non-zero probability of visiting the specified state.
    """

    b = [0] * size
    b[initialStates[tuple(state)]] = 1

    return b

#%% Solving difference equations

start = time.clock()

with open('console.txt','a') as file:
        file.write('N = {0} eta = {1} mu = {2}\n'.format(N,eta,mu))
        file.write('=========================\n')

p = 0.0
run = 1
initialStates = validStates(v_state)
Solutions = []
prob = []

if N != len(phi):
    p = 2
    with open('console.txt','a') as file:
        file.write('ERROR:\n The number of\n clonotypes does not\n match the length\n of the phi vector\n')
        file.write('=========================\n')

if N != len(v_state):
    p = 2
    with open('console.txt','a') as file:
        file.write('ERROR:\n The number of\n clonotypes does not\n match the state\n')
        file.write('=========================\n')

while p < 1+dp:

    run_start = time.clock()
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

#Storing Data
with open('Data.bin','wb') as file:
    data = (Solutions,prob,N,eta,mu,phi,gamma)
    pickle.dump(data,file)

#Writing to console
with open('console.txt','a') as file:
    file.write('=========================\n')
    file.write('Solutions stored\n')
    file.write('=========================\n')
    file.write('Total CPU time: {0:.3f}s\n'.format(time.clock() - start))
    file.write('=========================\n')
