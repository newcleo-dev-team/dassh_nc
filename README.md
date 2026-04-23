This is **newcleo's fork** of the original ANL DASSH repository (https://github.com/dassh-dev/dassh). 

Compared to the original ANL repository, this fork adds the following features:
- improved definition and update of coolant properties in the `Material` class;
- introduction of radially varying properties and a local approach to calculating the heat transfer coefficient;
- treatment of transverse mixing and mass exchange between subchannels.

The _Codes and Methods_ Department at newcleo manages and maintains this fork in collaboration with Politecnico di Torino. 

For information on this fork and how to contribute to it, please contact gabriele.ottino@newcleo.com.

For information about the original repository, please follow the recommendations provided in the ANL DASSH repository.  

# Ducted Assembly Steady State Heat Transfer Software (DASSH)

[![Build](https://github.com/dassh-dev/dassh/actions/workflows/ci.yml/badge.svg)](https://github.com/dassh-dev/dassh/actions)
[![codecov](https://codecov.io/gh/dassh-dev/dassh/branch/develop/graph/badge.svg)](https://app.codecov.io/gh/dassh-dev/dassh)
[![License](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)


The Ducted Assembly Steady State Heat Transfer Software (DASSH) is an open-source tool for calculating temperature and flow distributions in hexagonal, ducted assemblies comprised of wire-wrapped pin bundles. DASSH is intended for use during the design process to provide a rapid assessment of the flow and temperature distribution, especially when assembly designs are in their early stages and not fully developed.

## Installation
DASSH requires Python 3.6+. Detailed installation instructions can be found in the [user guide](https://github.com/dassh-dev/documents/blob/master/user_guide.pdf).

## Citing DASSH
If you use DASSH in your research, please consider citing the following conference paper, the theory manual, or the user guide.
* Milos Atz, Micheal A. Smith, Florent Heidet. “DASSH software for ducted assembly thermal hydraulics calculations – overview and benchmark”. Transactions of the American Nuclear Society 123 pp. 1673-1676 (2020). [URL](https://www.ans.org/pubs/transactions/article-49036/).
* Milos Atz, Micheal A. Smith, Florent Heidet, "Ducted Assembly Steady State Heat Transfer Software (DASSH) - Theory Manual", ANL/NSE-21/33, Argonne National Laboratory, 2021.
* Milos Atz, Micheal A. Smith, Florent Heidet, "Ducted Assembly Steady State Heat Transfer Software (DASSH) - User Guide", ANL/NSE-21/34, Argonne National Laboratory, 2021.

## Troubleshooting and reporting bugs
If you encounter issues installing or running DASSH or would like to report a bug, please reach out to the developer via `matz [at] anl [dot] gov`

## Documentation
* User guide: https://github.com/dassh-dev/documents/blob/master/user_guide.pdf
* Theory manual: https://github.com/dassh-dev/documents/blob/master/theory_manual.pdf

## License
DASSH is distributed under the [BSD-3](https://opensource.org/licenses/BSD-3-Clause) license.
