#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#%% Packages


import math
import pickle
import numpy as np
from random import uniform
from itertools import chain, combinations

#%% Global parameters


new_clone_is_soft = False
max_level_value = 179
mu_value = 1.0
n_mean_value = 10
gamma_value = 1.0
stimulus_value = [10 * gamma_value, 10 * gamma_value, 10 * gamma_value]
realisations = 1
time_max = 40
initial_state = [10, 10, 10]

distribution = np.zeros((max_level_value, max_level_value, max_level_value))

#%% Reading Samples and variables [Paper results]


SampleHolder = 0
probability_values = np.genfromtxt("../Samples/Matrices/Matrix-{}.csv".format(SampleHolder), delimiter=",")
dimension_value = probability_values.shape[0]

if SampleHolder < 3:
    if new_clone_is_soft:
        nu_value = np.genfromtxt("../../Samples/Nu-Matrices/Nu-Matrix-Soft.csv", delimiter=",")
    else:
        nu_value = np.genfromtxt("../Samples/Nu-Matrices/Nu-Matrix-Hard.csv", delimiter=",")
else:
    if new_clone_is_soft:
        nu_value = np.genfromtxt("../../Samples/Nu-Matrices/Nu-Matrix-Soft-(D).csv", delimiter=",")
    else:
        nu_value = np.genfromtxt("../../Samples/Nu-Matrices/Nu-Matrix-Hard-(D).csv", delimiter=",")
nu_value = nu_value * n_mean_value

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
    

def rate_list(state, probability, mu, nu, dimension, stimulus, max_level):
    """
    Creates a list of the birth and death rates of all clonotypes

    Parameters
    ----------
    state : List
        Number of cells per clonotype.
    probability : List
        Probability matrix.
    mu : float
        Single cell death rate.
    nu : numpy.ndarray
        Niche overlap matrix.
    dimension : int
        Number of clonotypes.
    stimulus : List
        Stimulus parameters.
    max_level : int
        Maximum level of the state space

    Returns
    -------
    rates : List
        List of all birth and death rates for all clonotypes.

    """

    rates = []
    
    if sum(state) < max_level:
        for i in range(dimension):
            rates.append(float(birth_rate(state, probability, i, dimension, nu, stimulus)))
            rates.append(float(state[i] * mu))
    else:
        for i in range(dimension):
            rates.append(float(state[i] * mu))

    return rates

#%% Gillespie Algorithm


total_realisations = 0
current_realisation = 0

while current_realisation < realisations:
    current_state = initial_state[:]
    current_time = 0.0
    while current_time <= time_max:
        r1 = uniform(0.0, 1.0)
        r2 = uniform(0.0, 1.0)
        alpha = rate_list(current_state, probability_values, mu_value, nu_value, dimension_value, stimulus_value, max_level_value)
        alpha_sum = float(sum(alpha))

        dt = -math.log(r1) / alpha_sum
        current_time += dt

        if len(alpha) == 2 * len(current_state):
            for current_rate in range(len(alpha)):
                if (sum(alpha[:current_rate]) / alpha_sum) <= r2 < (sum(alpha[:current_rate + 1]) / alpha_sum):
                    if current_rate % 2 == 0:
                        current_state[int(current_rate / 2)] += 1
                    else:
                        current_state[int(current_rate / 2)] -= 1
                        if current_state.count(0) > 0:
                            break
            else:
                continue
            break
        else:
            for current_rate in range(len(alpha)):
                if (sum(alpha[:current_rate]) / alpha_sum) <= r2 < (sum(alpha[:current_rate + 1]) / alpha_sum):
                    current_state[int(current_rate)] -= 1
                    if current_state.count(0) > 0:
                        break
            else:
                continue
            break
    else:
        distribution[current_state[0] - 1, current_state[1] - 1, current_state[2] - 1] += 1
        current_realisation += 1
    total_realisations += 1

#%% Storing results


# file = open('Parameters.bin', 'wb')
# parameters = (["dimension_value", "max_level_value", "mu_value", "gamma_value", "stimulus_value"], dimension_value, max_level_value, mu_value, gamma_value, stimulus_value)
# pickle.dump(parameters, file)
# file.close()

# file = open('Data.bin', 'wb')
# pickle.dump(distribution, file)
# file.close()
