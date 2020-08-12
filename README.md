# Calculating the distribution of absorption probabilities

This script calculates the distribution of absorption probabilities for ![equation](https://latex.codecogs.com/svg.latex?%5Ceta) clonotypes with starting states in ![equation](https://latex.codecogs.com/svg.latex?%5Cmathcal%7BA%7D%5E%7B0%7D) using first step analysis.

The birth and death rates are calculated using a matrix of self-pMHC sharing probabilities, for which values are sampled using Latin Hypercube Sampling, asuming that the probabilities are uniformly distributed.

The transition matrix for the embedded Markov process ![equation](https://latex.codecogs.com/svg.latex?%5Cmathbf%7BP%7D) is constructed, and it's associated matrix ![equation](https://latex.codecogs.com/svg.latex?%5Cmathbf%7BW%7D%3D%5Cleft%28%5Cmathbf%7BI%7D-%5Cmathbf%7BP%7D%20%5Cright%29%5E%7B-1%7D) is calculated.

The transition matrix from ![equation](https://latex.codecogs.com/svg.latex?%5Cmathcal%7BA%7D%5E%7B0%7D) to ![equation](https://latex.codecogs.com/svg.latex?%5Cmathcal%7BA%7D%5E%7B1%7D), named ![equation](https://latex.codecogs.com/svg.latex?%5Cmathbf%7BR%7D), is constructed and the product ![equation](https://latex.codecogs.com/svg.latex?%5Cmathbf%7BW%7D%5Cmathbf%7BR%7D) is calculated to obtain the desired absorption distributions for all starting states.

To make sure you have all the required dependencies in Anaconda create a new environment with the following commands.

```bash
conda create --name fs python=3.7.7
conda activate fs
pip install --upgrade pyDOE
```
