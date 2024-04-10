# FEniCSx examples at CISM 2024

## Introduction

## Authors

Jack S. Hale, University of Luxembourg.

Laura de Lorenzis, ETH Zurich.

Corrado Maurini, Sorbonne Université.

## Developer instructions

### Building

To build this notebook run:

    docker run -v $(pwd):/shared -w /shared -ti dolfinx/lab:nightly
    pip install -r requirements-docs.txt
    jupyter-book build .

### Linting and formatting

To lint and format using ruff:

    pip install ruff
    ruff check .
    ruff format .

### Converting .ipynb to .py

This repository stores stores demos in percent-formatted .py files that can
then be converted to e.g. notebooks on demand. To convert legacy notebooks e.g.
`notebook.ipynb` to this format automatically use:

    pip install jupytext
    jupytext --to py notebook.ipynb

which will create a file `notebook.py` which you can then edit with any text
editor. For more information see
[here](https://jupytext.readthedocs.io/en/latest/formats-scripts.html#).

### Credits

This work includes elements from [Computational fracture mechanics examples
with FEniCSx](https://github.com/newfrac/fenicsx-fracture) under the terms of
the BSD license.

This work includes elements from [NewFrac FEniCSx
training](https://newfrac.gitlab.io/newfrac-fenicsx-training/) under the terms
of the MIT license.
