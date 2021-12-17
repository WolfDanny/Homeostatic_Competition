# Python scripts used to calculate stochastic descriptors of the homeostatic competition model 

This repository contains the codes used to calculate the QSD, absorption distribution, distribution of divisions, and mean time to extinction for the homeostatic competition model.

## Scripts included in ths repository

* `Homeostatic-QSD.py`: used to approximate the QSD of the process using 2 approximating processes where there is no extinction. 
* `Homeostatic-QSD-Gillespie.py`: used to approximate the QSD by simulating the stochastic process multiple times using the Gillespie algorithm.
* `Absorption-distribution.py`: used to calculate the distribution of absorption probabilities from all starting states using a first step argument.
* `Division-distribution.py`: used to calculate the distribution of divisions before extinction from all starting states using first step analysis.
* `Mean-time.py`: used to calculate the mean time until the first extinction event of the competition process.

The `Samples` directory contains the competition probabilities for all the competition scenarios considered in the paper, as well as the values for the mean niche overlap.

## Python environment
To make sure you have all the required dependencies in Anaconda, create and activate a new environment use the following commands:

```bash
conda create --name fs python=3.7.11 matplotlib=3.4.3 numpy=1.21.2 scipy
conda activate fs
```

## Using the python scripts

Due to space limitations only the results of `Homeostatic-QSD.py`, `Homeostatic-QSD-Gillespie.py`, and `Mean-time.py` are included in `Results`

1. Run `QSD-plots-established.py` in `Results/QSD` to generate the figures for the QSD of the established clonotypes and `Means.bin`.
2. Run `QSD-plots.py` in `Results/QSD` to generate the figures for the QSD of the three clonotypes and `Truncated_levels.bin`.
3. Run `Absorption-distribution.py`, and `Division-distribution.py` to generate the data.
4. Run `AD-plots.py`, `AD-plots-path.py`, and `DD-plots.py` to generate the figures for the absorption distribution and the distribution of divisions.
