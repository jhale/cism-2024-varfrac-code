import dolfinx
import ufl
from mpi4py import MPI

import numpy as np

import basix
import basix.ufl
import dolfinx.fem as fem
import dolfinx.fem.petsc  # noqa: F401
import dolfinx.io as io
import dolfinx.mesh as mesh
import dolfinx.plot as plot
import ufl
import sys

sys.path.append("../utils/")

from meshes import generate_bar_mesh

msh, mt, ft = generate_bar_mesh(Lx=1.0, Ly=0.3, lc=0.05)

import pyvista  # noqa: E402

pyvista.start_xvfb(wait=0.1)
pyvista.set_jupyter_backend("static")

vtk_mesh = plot.vtk_mesh(msh)
grid = pyvista.UnstructuredGrid(*vtk_mesh)
plotter = pyvista.Plotter()
plotter.add_mesh(grid, show_edges=True)
plotter.camera_position = "xy"
if not pyvista.OFF_SCREEN:
    plotter.show()
