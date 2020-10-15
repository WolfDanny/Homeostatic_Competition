# Calculating the distribution of absorption probabilities

This script calculates the distribution of absorption probabilities for ![equation](https://latex.codecogs.com/svg.latex?%5Ceta) clonotypes with starting states in ![equation](https://latex.codecogs.com/svg.latex?%5Cmathcal%7BA%7D%5E%7B0%7D) using first step analysis, denoted ![equation](https://latex.codecogs.com/svg.latex?%5Cmathbf%7BU%7D).

The birth and death rates are calculated using a matrix of self-pMHC sharing probabilities, for which values are sampled using Latin Hypercube Sampling, assuming that the probabilities are uniformly distributed.

The transition matrix for the embedded Markov process ![equation](https://latex.codecogs.com/svg.latex?%5Cmathbf%7BP%7D) is constructed, and the inverse of it's associated matrix, ![equation](https://latex.codecogs.com/svg.latex?%5Cmathbf%7BW%7D%5E%7B-1%7D%3D%5Cmathbf%7BI%7D-%5Cmathbf%7BP%7D), is calculated.

The transition matrix from ![equation](https://latex.codecogs.com/svg.latex?%5Cmathcal%7BA%7D%5E%7B0%7D) to ![equation](https://latex.codecogs.com/svg.latex?%5Cmathcal%7BA%7D%5E%7B1%7D), denoted ![equation](https://latex.codecogs.com/svg.latex?%5Cmathbf%7BR%7D), is constructed and the matrix equation ![equation](https://latex.codecogs.com/svg.latex?%5Cleft%28%5Cmathbf%7BI%7D-%5Cmathbf%7BP%7D%5Cright%29%5Cmathbf%7BU%7D%3D%5Cmathbf%7BR%7D) is solved for ![equation](https://latex.codecogs.com/svg.latex?%5Cmathbf%7BU%7D) using the tri-diagonal by block structure of ![equation](https://latex.codecogs.com/svg.latex?%5Cmathbf%7BI%7D-%5Cmathbf%7BP%7D).

When running this script include `Samples.bin` and `Truncated_levels.bin` in the same directory, and replace `SampleHolder` with the desired sample number for which the absorption distribution is to be calculated.