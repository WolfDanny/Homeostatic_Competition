#%% Packages

from pyDOE import *
import pickle

#%% Variables

dimension = 3
strata = 20

#%% LHS sampling

samples = lhs((2 ** dimension) - dimension - 1, strata)

#%% Storing data

file = open('Samples.bin','wb')
pickle.dump((dimension,strata,samples), file)
file.close()
