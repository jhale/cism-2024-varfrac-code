# FEniCSx examples at CISM 2024

## Introduction

This repository contains source code for the computational examples presented
at the 9th CISM-ECCOMAS Advanced Course on Variational Fracture Mechanics and
Phase-Field Models taking place between July 1st 2014 and July 5th 2024.

The built book can be read at https://jhale.github.io/cism-2024-varfrac-code/.
The git repository can be found at https://github.com/jhale/cism-2024-varfrac-code/.

## Updates

- 11/2024: Update for DOLFINx `v0.9.0`.

## Installation instructions

The *recommended* approach for installing the software is using Docker. By
using Docker we ensure a consistent software environment across all
participants and minimal use of the wireless network. As alternatives, we also
provide instructions for [Google Colab](https://colab.research.google.com) and
[Anaconda](https://www.anaconda.com/download) (`conda`).

### Docker

1. Install Docker Desktop on your system following the instructions
   [here](https://www.docker.com/products/docker-desktop/).

2. Pull the DOLFINx laboratory image:

       docker pull dolfinx/lab:v0.9.0r1

3. (macOS, Linux). Start a DOLFINx laboratory container using a Unix-like shell:

       mkdir ~/cism-varfrac-course
       cd ~/cism-varfrac-course
       docker run -ti -v "$(pwd)":/shared -p 8888:8888 -w /shared dolfinx/lab:v0.9.0r1 

4. (Windows Powershell). Start a DOLFINx laboratory container using
   Powershell: 
       
       mkdir ~/cism-varfrac-course
       cd ~/cism-varfrac-course
       docker run -ti -v "${pwd}:/shared" -p 8888:8888 -w /shared dolfinx/lab:v0.9.0r1

4. (Windows cmd). Start a DOLFINx laboratory container using Windows cmd.
       
       mkdir %HOMEPATH%\cism-varfrac-course
       cd %HOMEPATH%\cism-varfrac-course
       docker run -ti -v "%cd%":/shared -p 8888:8888 -w /shared dolfinx/lab:v0.9.0r1

5. A URL e.g.
   `http://127.0.0.1:8888/lab?token=544f7380ab06eb1d175d8c2b35a362e7fd7a29471b56818c`
   will be printed to the terminal. Copy this to a web browser. You should see
   the Jupyter notebook environment open.

7. Click `File > New > Notebook` in the top menu. Use the Python 3 (ipykernel)
   kernel by pressing `Select`. In the first notebook cell type:

       import dolfinx
       import pyvista
       import gmsh

   and then press `Shift + Enter` to execute the cell. You should receive no
   errors (e.g. `ModuleNotFoundError`).

### Conda (macOS and Linux)

1. Download the [Anaconda Python distribution](https://www.anaconda.com/download).

2. Create and activate a new `conda` enviroment:

       mkdir ~/cism-varfrac-course
       cd ~/cism-varfrac-course
       curl -o environment.yml https://raw.githubusercontent.com/jhale/cism-varfrac-code/main/environment.yml
       conda env create -f environment.yml
       conda activate fenicsx-cism-2024

3. Launch a Jupyter lab environment in your browser:

       jupyter-lab .

4. Click `File > New > Notebook` in the top menu. Use the Python 3 (ipykernel)
   kernel by pressing `Select`. In the first notebook cell type:

       import dolfinx
       import pyvista
       import gmsh
   
   and then press `Shift + Enter` to execute the cell. You should receive no
   errors (e.g. `ModuleNotFoundError`). If macOS asks about accepting incoming
   network connections, click Allow.

### Google Colab

1. Go to [Google Colab](https://colab.research.google.com) and create a new
   notebook. We will use the [FEM on Colab](https://fem-on-colab.github.io)
   project to install FEniCSx. Copy and paste into a new notebook cell:

       try:
           import dolfinx
       except ImportError:
           !wget "https://fem-on-colab.github.io/releases/fenicsx-install-real.sh" -O "/tmp/fenicsx-install.sh" && bash "/tmp/fenicsx-install.sh"
       import dolfinx

    and press `Shift+Enter`. You should see output from the install process and
    no errors.

## Getting up and running

We recommend cloning the repository to run the notebooks using `git`.

From a terminal inside Jupyterlab:

    cd /shared
    git clone https://github.com/jhale/cism-2024-varfrac-code.git
    cd cism-2024-varfrac-code

If `git` does not work you can also download the current version using:

    cd /shared
    curl -L -o main.tar.gz https://github.com/jhale/cism-2024-varfrac-code/archive/refs/heads/main.tar.gz
    tar -xvf main.tar.gz 
    cd cism-2024-varfrac-code-main

Users using Google Colab should paste the following content into a cell:

    from google.colab import drive
    drive.mount('/content/drive')
    %cd "/content/drive/MyDrive/Colab Notebooks" 
    !git clone https://github.com/jhale/cism-2024-varfrac-code.git
    %cd cism-2024-varfrac-code

To open the notebook in Colab navigate using Google Drive to `Colab Notebooks/`
and double click on it. 

## Authors

- Jack S. Hale, University of Luxembourg.
- Laura de Lorenzis, ETH Zurich.
- Corrado Maurini, Sorbonne Université.

## Credits

This work includes elements from [Computational fracture mechanics examples
with FEniCSx](https://github.com/newfrac/fenicsx-fracture) under the terms of
the BSD license.

This work includes elements from [NewFrac FEniCSx
training](https://newfrac.gitlab.io/newfrac-fenicsx-training/) under the terms
of the MIT license.

## Developer instructions

### Building

To build this Jupyter book run:

    docker run -v $(pwd):/shared -w /shared -ti --entrypoint /bin/bash dolfinx/lab:v0.9.0r1 
    pip install -r requirements-docs.txt
    export PYVISTA_OFF_SCREEN=true
    export PYVISTA_JUPYTER_BACKEND=html
    jupyter-book build -W .

and then on the host:

    open _build/html/index.html

On ARM it is necessary to install `pyvista` from a custom binary wheel, see
below.

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

### pyvista on ARM

To install pyvista in ARM docker:

    python3 -m pip install "https://github.com/finsberg/vtk-aarch64/releases/download/vtk-9.2.6-cp310/vtk-9.2.6.dev0-cp310-cp310-linux_aarch64.whl"
    python3 -m pip install pyvista
