# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
# ---

# +
import sys

import numpy as np

import basix
import basix.ufl
import dolfinx
import dolfinx.fem as fem
import dolfinx.fem.petsc
import dolfinx.plot as plot
import ufl

sys.path.append("../utils/")

from meshes import generate_bar_mesh

# By dimensional analysis
sigma_c = 1.0  #
Lx = 1.0  # Size of domain in x-direction
Ly = 0.1  # Size od domain in y-direction

# Define free parameters. Table 1 Zelosi and Maurini, first row.
G_c = 1.0  # Fracture toughness.
E_0 = 1.0  # Young's modulus.
ell = 0.025  # Regularisation length scale.

# Additional parameters
nu_0 = 0.3  # Poisson's ratio.

# Computational parameters
num_steps = 50  # Number of load steps
lc = ell / 5.0  # Characteristic mesh size

# Derived quantities
mu_0 = E_0 / (2 * (1 + nu_0))
kappa_0 = E_0 / (2 * (1 - nu_0))
w_1 = G_c / (np.pi * ell)
gamma_traction = (2 * w_1 * E_0) / (sigma_c**2)
t_c = sigma_c / E_0
t_f = gamma_traction * t_c
t_star = 2 * np.pi * ell / Lx * w_1 / sigma_c

load_elastic = np.linspace(0, 0.95 * t_c, 10)[:-1]
load_damage = np.linspace(0.95 * t_c, 1.3 * np.max([t_star, t_c]), num_steps)
loads = np.concatenate((load_elastic, load_damage)) / t_c

msh, mt, ft = generate_bar_mesh(Lx=Lx, Ly=Ly, lc=lc)

import pyvista  # noqa: E402

pyvista.start_xvfb(wait=0.1)
pyvista.set_jupyter_backend("static")

vtk_mesh = plot.vtk_mesh(msh)
grid = pyvista.UnstructuredGrid(*vtk_mesh)
plotter = pyvista.Plotter()
plotter.add_mesh(grid, show_edges=True)
plotter.camera_position = "xy"
# if not pyvista.OFF_SCREEN:
#    plotter.show()

element_u = basix.ufl.element("Lagrange", msh.basix_cell(), degree=1, shape=(msh.geometry.dim,))
V_u = fem.functionspace(msh, element_u)

element_alpha = basix.ufl.element("Lagrange", msh.basix_cell(), degree=1)
V_alpha = fem.functionspace(msh, element_alpha)

u = fem.Function(V_u, name="displacement")
alpha = fem.Function(V_alpha, name="damage")

alpha_lb = dolfinx.fem.Function(V_alpha, name="lower bound")
alpha_ub = dolfinx.fem.Function(V_alpha, name="upper bound")

dx = ufl.Measure("dx", domain=msh)
ds = ufl.Measure("ds", domain=msh, subdomain_data=ft)

print(dir(ft))
print(ft.find(9))
