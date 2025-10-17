This folder contains the evaluation of the density-to-enthalpy conversion method, which is based on the derivation of a fitting polynomial. It is only present in the current branch because it motivates the implementation choices regarding density-to-enthalpy conversion. 

In particular, this folder contains the following files:
- _density2enthalpy.ipynb_: describes the approach and presents results for each coolant;
- _auxiliary.py_: contains functions for accuracy and time effort evaluation, and for database generation;
- __common.py_: contains global variables.

The folder also contains the _data_ subfolder, which stores tables used for polynomial fitting and for reference. 