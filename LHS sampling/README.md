# Sampling values for the probability of sharing self pMHCs

This script uses LHS sampling, using the pyDOE implementation found here https://pythonhosted.org/pyDOE/index.html, to sample ![equation](https://latex.codecogs.com/svg.latex?2%5E%7B%5Ceta%7D-%5Ceta-1) times the interval [0,1], divided in 20 strata. The samples are sorted in decreasing order by their first element.

The samples are saved on a binary file and are used by the other scripts to calculate the matrix containing the values of ![equation](https://latex.codecogs.com/svg.latex?p_%7Bij%7D%5E%7Bk%7D).