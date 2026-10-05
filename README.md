# Ducted Assembly Steady State Heat Transfer Software (DASSH_nc) - forked from ANL DASSH
[![License](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)

**DASSH_nc** is the <em>new</em>cleo's fork of the ANL DASSH repository (https://github.com/dassh-dev/dassh).

[![Build](https://github.com/dassh-dev/dassh/actions/workflows/ci.yml/badge.svg)](https://github.com/dassh-dev/dassh/actions)
[![License](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)

|    <!-- -->    |        <!-- -->    |
|----------------|--------------------|
| Author: | Francesco Pepe |
| Contributor: | Gabriele Ottino |

DASSH-nc is a <em>new</em>cleo fork of the ANL DASSH repository (https://github.com/dassh-dev/dassh). 

The Ducted Assembly Steady State Heat Transfer Software (**DASSH**) is an open-source tool for calculating temperature and flow distributions in hexagonal, ducted assemblies comprised of wire-wrapped pin bundles. DASSH is intended for use during the design process to provide a rapid assessment of the flow and temperature distribution, especially when assembly designs are in their early stages and not fully developed.

Compared to the original ANL repository, this fork adds the following features:

- improved definition and update of coolant properties in the Material class;
- introduction of radially varying properties and a local approach to calculating the heat transfer coefficient;
- treatment of transverse mixing and mass exchange between subchannels.

The _Codes and Methods Department_ at <em>new</em>cleo manages and maintains this fork in collaboration with Politecnico di Torino.


## Project Structure

The project is organized according to the following folder structure:



```text

<dassh parent folder>
├── dassh/
├── tests/
├── LICENSE
├── README.md
├── requirements.txt
└── setup.py
```

* `dassh`: contains all modules, classes and methods implemented in DASSH;

* `tests`: collection of tests used to verify the correct implementation.



## Dependencies
To run the code, the following dependencies must be satisfied:

* `python>=3.6`

* `pandas>=1.1.5`

* `configobj==5.0.6`

* `pytest==6.2.4`

* `pytest-cov==2.12.0`

* `matplotlib>=3.2`

* `numpy>=1.18`

* `codecov==2.1.11`

* `dill==0.3.4`

* `packaging>=20.0`

* `sympy==1.13.3`



## Installation
DASSH requires Python 3.6+. Detailed installation instructions can be found in the [user guide](https://github.com/dassh-dev/documents/blob/master/user_guide.pdf).

To install **DASSH_nc**, clone its repo and execute the following command inside the base folder:

    pip install .


## Documentation

* User guide: https://github.com/dassh-dev/documents/blob/master/user_guide.pdf

* Theory manual: https://github.com/dassh-dev/documents/blob/master/theory_manual.pdf


## How to cite

If you use **DASSH_nc** in your research, please consider citing the following items:

* Milos Atz, Micheal A. Smith, Florent Heidet. “DASSH software for ducted assembly thermal hydraulics calculations – overview and benchmark”. Transactions of the American Nuclear Society 123 pp. 1673-1676 (2020). [URL](https://www.ans.org/pubs/transactions/article-49036/).
* Milos Atz, Micheal A. Smith, Florent Heidet, "Ducted Assembly Steady State Heat Transfer Software (DASSH) - Theory Manual", ANL/NSE-21/33, Argonne National Laboratory, 2021.
* Milos Atz, Micheal A. Smith, Florent Heidet, "Ducted Assembly Steady State Heat Transfer Software (DASSH) - User Guide", ANL/NSE-21/34, Argonne National Laboratory, 2021.

## Troubleshooting and reporting bugs
If you encounter issues installing or running DASSH-nc or would like to report a bug, please reach out to the developer via _francesco.pepe@newcleo.com_. 

## Documentation
* User guide: https://github.com/dassh-dev/documents/blob/master/user_guide.pdf
* Theory manual: https://github.com/dassh-dev/documents/blob/master/theory_manual.pdf

User and Theory manuals cover the original DASSH. 
Please refer to [#6](https://github.com/newcleo-dev-team/dassh_nc/issues/6) for documentation updates regarding the modifications implemented in this fork. 

## License

DASSH_nc is distributed under the [BSD-3](https://opensource.org/licenses/BSD-3-Clause) license.
