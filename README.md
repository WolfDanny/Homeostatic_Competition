# Python scripts used to calculate stochastic descriptors of the homeostatic competition model 

This repository contains the codes used to calculate the QSD, absorption distribution, distribution of divisions, and mean time to extinction for the homeostatic competition model.

`Homeostatic-QSD.py` is used to approximate the QSD of the process using 2 approximating processes where there is no extinction. `Homeostatic-QSD-Gillespie.py` is used to approximate the QSD by simulating the stochastic process multiple times using the Gillespie algorithm. `Absorption-distribution.py` is used to calculate the distribution of absorption probabilities from all starting states using a first step argument. `Division-distribution.py` is used to calculate the distribution of divisions before extinction from all starting states using first step analysis. `Mean-time.py` calculates the mean time to extinction of the competition process.

The `Samples` directory contains the competition probabilities for all the competition scenarios considered in the paper, as well as the values for the mean niche overlap.

To make sure you have all the required dependencies in Anaconda, create a new environment use the following commands:

```bash
conda create --name fs python=3.7.11 matplotlib=3.4.3 numpy=1.21.2 scipy=1.7.1
conda activate fs
```
