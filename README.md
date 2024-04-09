# cism-varfrac-code
## FEniCSx examples at CISM 2024

### Introduction

### Authors

Jack S. Hale, University of Luxembourg.

Laura de Lorenzis, ETH Zurich.

Corrado Maurini, Sorbonne Université.

### Developer instructions

#### Building

To build this notebook run:

    docker run -v $(pwd):/shared -w /shared -ti dolfinx/dolfinx:nightly
    pip install jupyter-book
    jupyter-book build .

#### Linting and formatting

To lint and format using ruff:

    pip install ruff
    ruff check .
    ruff format .

### Credits

This work includes elements from [Computational fracture mechanics examples
with FEniCSx](https://github.com/newfrac/fenicsx-fracture) under the terms of
the BSD license.

This work includes elements from [NewFrac FEniCSx
training](https://newfrac.gitlab.io/newfrac-fenicsx-training/) under the terms
of the MIT license.
