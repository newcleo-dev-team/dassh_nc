This folder contains the time effort evaluation of three methods for solving linear systems:
1) `numpy.linalg.solve`
2) `scipy.sparse.linalg.solve`
3) the **Greene's method**.

In detail, this folder contains the following items:
- _solver_evaluation.ipynb_: computes average time effort for each method;
- *_common.py*: contains common variables and functions;
- *power.csv*: power distribution for the test case;
- the subfolders *greene*, *numpy*, *scipy*, containing input files to be run by dassh.