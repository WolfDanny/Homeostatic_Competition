#%% Packages

from pyDOE import *
import numpy as np
import pickle

#%% Variables

dimension = 3
strata = 20

#%% LHS sampling

samples = lhs((2 ** dimension) - dimension - 1, samples=strata, criterion='center')

sort_column = 0
samples = samples[np.argsort(samples[:, sort_column])]

samples = samples[::-1]  # Reverse the order

#%% Storing data

file = open('Samples.bin', 'wb')
pickle.dump((dimension, strata, samples), file)
file.close()
