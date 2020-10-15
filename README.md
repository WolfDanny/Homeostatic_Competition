#Python scripts used to calculate stochastic descriptors of the homeostatic competition model 

This repository contains the codes used to calculate the QSD, distribution of absorptions, and distribution of divisions for the homeostatic competition model.

`LHS-sampling.py` is used to sample for the probabilities of sharing self pMHCs. `Homeostatic-QSD.py` is used to approximate the QSD of the process using 2 approximating processes where there is no extinction. `FS-absorption-pobability.py` is used to calculate the distribution of absorption probabilities from all starting states using first step analysis. `FS-division-pobability.py` is used to calculate the distribution of divisions before extinction from all starting states using first step analysis.

The files `Samples.bin` and `Truncated_levels.bin` used can be found in the `Samples` directory.

To make sure you have all the required dependencies in Anaconda, activate or create a new environment with python 3.7.7 and pyDOE installed. To create a new environment with these conditions use the following commands:

```bash
conda create --name fs python=3.7.7
conda activate fs
pip install --upgrade pyDOE
```
