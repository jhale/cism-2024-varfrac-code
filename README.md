# cism-varfrac-code
## FEniCSx examples at CISM 2024

### Introduction

### Building

To build this notebook run:

    docker run -v $(pwd):/shared -w /shared -ti dolfinx/dolfinx:nightly
    pip install jupyter-book
    jupyter-book build .

### Authors

Jack S. Hale, University of Luxembourg.

Laura de Lorenzis, ETH Zurich.

Corrado Maurini, Sorbonne Université.

### Credits

This work includes elements from [Computational fracture mechanics examples
with FEniCSx](https://github.com/newfrac/fenicsx-fracture) under the terms of
the BSD license.

This work includes elements from [NewFrac FEniCSx
training](https://newfrac.gitlab.io/newfrac-fenicsx-training/) under the terms
of the MIT license.
