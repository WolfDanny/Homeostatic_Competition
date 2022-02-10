[![DOI](https://zenodo.org/badge/282223835.svg)](https://zenodo.org/badge/latestdoi/282223835) [![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
# Python scripts used to calculate stochastic descriptors of the homeostatic competition model

This repository contains the codes used to calculate the QSD, absorption distribution, distribution of divisions, and mean time to extinction for the homeostatic competition model.

## Scripts included in ths repository

* `Homeostatic_QSD.py`: used to approximate the QSD of the process using 2 approximating processes where there is no extinction.
* `Homeostatic_QSD_Gillespie.py`: used to approximate the QSD by simulating the stochastic process multiple times using the Gillespie algorithm.
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

Due to space limitations only the results of `Homeostatic_QSD.py`, `Homeostatic_QSD_Gillespie.py`, and `Mean-time.py` are included in `Results`

1. Run `QSD-plots-established.py` in `Results/QSD` to generate the figures for the QSD of the established clonotypes and `Means.bin`.
2. Run `QSD-plots.py` in `Results/QSD` to generate the figures for the QSD of the three clonotypes and `Truncated_levels.bin`.
3. Run `Absorption-distribution.py`, and `Division-distribution.py` to generate the data.
4. Run `Results/Absorption distribution/AD-plots.py`, `Results/Absorption distribution/AD-plots-path.py`, and `Results/Division distribution/DD-plots.py` to generate the figures for the absorption distribution and the distribution of divisions.

## Using `Homeostatic_QSD.py`, `Homeostatic_QSD_Gillespie.py`, and `Mean-time.py`

When using `Homeostatic_QSD.py`, `Homeostatic_QSD_Gillespie.py`, and `Mean-time.py` they must be run several times to generate a complete dataset.

1. `Mean-time.py` must be run first to identify the appropriate value for `time_max` in `Homeostatic_QSD_Gillespie.py`. After all the data is produced `Results/Mean time to extinction/MT-analysis.py` can be run.
   - For `clones = 3` the code needs to be run for all combinations of the following parameters:
     - `new_clone_is_soft` has to be set to `True` or `False`.
     - `sample_value` has to be set to `0`, `1`, `2`, or `3`.
   - For `clones = 2` the code only needs to be run once (other parameters are not used in this case).
2. `Homeostatic_QSD.py` must be run for all combinations of the following parameters:
   - For `clones = 3` the code needs to be run for all combinations of the following parameters:
     - `new_clone_is_soft` has to be set to `True` or `False`.
     - `sample_value` has to be set to `0`, `1`, `2`, or `3`.
     - `model_value` has to be set to `0`, or `1`.
   - For `clones = 2` the code only needs to be run once (other parameters are not used in this case).
3. `Homeostatic_QSD_Gillespie.py` must be run similarly to `Homeostatic_QSD.py` with the addition of the following parameters:
   - `realisations`: The number of realisations (note that the actual number of simulations run will be higher since we only consider those with no extinction events).
   - `time_max`: Time at which the simulation ends (use the results of `Results/Mean time to extinction/MT-analysis.py` to determine an appropriate value).
