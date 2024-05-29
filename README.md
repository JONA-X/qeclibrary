# QEClib

## Features

Codes that are currently supported:
* Rotated surface codes

Lattice that are currently supported:
* Square lattice

Operations that are currently supported:
* Add the circuit for extracting syndromes
* Perform logical X and Z gates
* Perform transversal logical Hadamard gates (which changes the type of stabilizers and logical operators and thus also the boundaries. No rotating back yet)
* Measure logical X and Z operators
* Perform logical quantum state tomography
* Split one logical qubit into two


## Demo notebooks

Demo notebooks are located in the `notebooks` folder:

* Quantum memory experiment for rotated surface codes of increasing distance (without decoding): [memory_exp.ipynb](notebooks/memory_exp.ipynb)
* Logical quantum state tomography for two d=3 rotated surface codes: [two_qubit_logical_QST.ipynb](notebooks/two_qubit_logical_QST.ipynb)
* Split one 3x7 rotated surface code qubit into two d=3 codes: [split.ipynb](notebooks/split.ipynb)