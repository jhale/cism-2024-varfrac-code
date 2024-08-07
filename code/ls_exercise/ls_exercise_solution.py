# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.2
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# + [markdown]
# # Solution: Implement the LS model from the AT1 state
#
# *Authors:*
# - Jack S. Hale (University of Luxembourg)
# - Corrado Maurini (Sorbonne Université)
#
# This notebook contains a slightly modified version of the AT1 gradient damage
# tutorial. Modify it so that it expresses the LS model using the 3rd
# parameterisation of Table 1 and case b) the intermediate-length bar in Figure
# 7:
#
# - Stability and crack nucleation in variational phase-field models of
# fracture: effects of length-scales and stress multi-axiality. Zolesi and
# Maurini 2024. Preprint. https://hal.sorbonne-universite.fr/hal-04552309
#
# ## Preamble
#
# We begin by importing the required Python modules.
#
# +
import sys

from mpi4py import MPI
from petsc4py import PETSc

import matplotlib.pyplot as plt
import numpy as np

import basix
import dolfinx
import dolfinx.fem.petsc
import ufl
from dolfinx import fem, io, la, mesh, plot

sys.path.append("../utils/")

import pyvista
from evaluate_on_points import evaluate_on_points
from petsc_problems import SNESProblem
from plots import plot_damage_state
from pyvista.utilities.xvfb import start_xvfb

start_xvfb(wait=0.5)

# + [markdown]
# ## Mesh
#
# We define the mesh using the built-in DOLFINx mesh generation functions for
# simply geometries.
# +
L = 1.0
H = 0.3
ell_ = 0.1
cell_size = ell_ / 6

nx = int(L / cell_size)
ny = int(H / cell_size)

comm = MPI.COMM_WORLD
msh = mesh.create_rectangle(
    comm, [(0.0, 0.0), (L, H)], [nx, ny], cell_type=mesh.CellType.quadrilateral
)
ndim = msh.geometry.dim

topology, cell_types, geometry = plot.vtk_mesh(msh)
grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)
plotter = pyvista.Plotter()
plotter.add_mesh(grid, show_edges=True, show_scalar_bar=True)
plotter.view_xy()
plotter.add_axes()
plotter.set_scale(5, 5)
if not pyvista.OFF_SCREEN:
    plotter.show()

# + [markdown]
# ## Setting the stage
#
# We setup the finite element space, the states, the bound constraints on the
# states and UFL measures.
#
# We use (vector-valued) linear Lagrange finite elements on quadrilaterals for
# displacement and damage.
# +
element_u = basix.ufl.element("Lagrange", msh.basix_cell(), degree=1, shape=(msh.geometry.dim,))
V_u = fem.functionspace(msh, element_u)

element_alpha = basix.ufl.element("Lagrange", msh.basix_cell(), degree=1)
V_alpha = fem.functionspace(msh, element_alpha)

# Define the state
u = fem.Function(V_u, name="displacement")
alpha = fem.Function(V_alpha, name="damage")

# Domain measure.
dx = ufl.Measure("dx", domain=msh)


# + [markdown]
# ### Boundary conditions
# We impose Dirichlet boundary conditions on the displacement and the damage
# field on the appropriate parts of the boundary.
#
# We do this using predicates. DOLFINx will pass an array of the midpoints of
# all facets (edges) as an argument `x` with shape `(3, num_edges)` to our
# predicate. The predicate we define must return an boolean array of shape
# `(num_edges)` containing `True` if the edge is on the desired boundary, and
# `False` if not.
# +
def right(x):
    return np.isclose(x[0], L)


def left(x):
    return np.isclose(x[0], 0.0)


# + [markdown]
# The function `mesh.locate_entities_boundary` calculates the indices of the
# edges on the boundary defined by our predicate.
# +
fdim = msh.topology.dim - 1

left_facets = mesh.locate_entities_boundary(msh, fdim, left)
right_facets = mesh.locate_entities_boundary(msh, fdim, right)

# + [markdown]
# The function `fem.locate_dofs_topological` calculates the indices of the
# degrees of freedom associated with the edges. This is the information the
# assembler will need to apply Dirichlet boundary conditions.
# +
left_boundary_dofs_ux = fem.locate_dofs_topological(V_u.sub(0), fdim, left_facets)
right_boundary_dofs_ux = fem.locate_dofs_topological(V_u.sub(0), fdim, right_facets)

# + [markdown]
# Using `fem.Constant` will allow us to update the value of the boundary
# condition applied in the pseudo-time loop.
# +
u_D = fem.Constant(msh, 0.5)
bc_ux_left = fem.dirichletbc(0.0, left_boundary_dofs_ux, V_u.sub(0))
bc_ux_right = fem.dirichletbc(u_D, right_boundary_dofs_ux, V_u.sub(0))

bcs_u = [
    bc_ux_left,
    bc_ux_right,
]

# + [markdown]
# and similarly for the damage field.
# +
left_boundary_dofs_alpha = fem.locate_dofs_topological(V_alpha, fdim, left_facets)
right_boundary_dofs_alpha = fem.locate_dofs_topological(V_alpha, fdim, right_facets)

bc_alpha_left = fem.dirichletbc(0.0, left_boundary_dofs_alpha, V_alpha)
bc_alpha_right = fem.dirichletbc(0.0, right_boundary_dofs_alpha, V_alpha)

bcs_alpha = [bc_alpha_left, bc_alpha_right]

# + [markdown]
# ## Variational formulation of the problem
# ### Constitutive model
#
# To implement the S-LS you will need to define new parameters and modify the
# model functions appropriately.
#
# +
E, nu = (
    fem.Constant(msh, dolfinx.default_scalar_type(1.0)),
    fem.Constant(msh, dolfinx.default_scalar_type(0.3)),
)
Gc = fem.Constant(msh, dolfinx.default_scalar_type(1.0))
ell = fem.Constant(msh, dolfinx.default_scalar_type(ell_))
c_w = fem.Constant(msh, dolfinx.default_scalar_type(np.pi))
gamma = 2.0
sigma_c = np.sqrt((2.0 * Gc.value * E.value) / (np.pi * ell.value * gamma))
eps_c = sigma_c / E.value


def w(alpha):
    """Dissipated energy function as a function of the damage"""
    return 1.0 - (1.0 - alpha) ** 2


def a(alpha, k_ell=1.0e-6):
    """Stiffness modulation as a function of the damage"""
    return (1 - w(alpha)) / (1.0 + (gamma - 1.0) * w(alpha))


def eps(u):
    """Strain tensor as a function of the displacement"""
    return ufl.sym(ufl.grad(u))


def sigma_0(eps):
    """Stress tensor of the undamaged material as a function of the strain"""
    mu = E / (2.0 * (1.0 + nu))
    lmbda = E * nu / (1.0 - nu**2)
    return 2.0 * mu * eps + lmbda * ufl.tr(eps) * ufl.Identity(ndim)


def sigma(eps, alpha):
    """Stress tensor of the damaged material as a function of the displacement and the damage"""
    return a(alpha) * sigma_0(eps)


# + [markdown]
# ### Energy functional and its derivatives
#
# We use the `ufl` package of FEniCS to define the energy functional. The
# residual (first Gateaux derivative of the energy functional) and Jacobian
# (second Gateaux derivative of the energy functional) can then be derived
# through automatic symbolic differentiation using `ufl.derivative`.
# +
f = fem.Constant(msh, PETSc.ScalarType((0.0, 0.0)))
elastic_energy = 0.5 * ufl.inner(sigma(eps(u), alpha), eps(u)) * dx
dissipated_energy = (
    Gc / c_w * (w(alpha) / ell + ell * ufl.inner(ufl.grad(alpha), ufl.grad(alpha))) * dx
)
external_work = ufl.inner(f, u) * dx
total_energy = elastic_energy + dissipated_energy - external_work

# + [markdown]
# To block rigid bode modes in a simple but effective way, we add a very weak
# elastic foundation.
# +
k_springs = fem.Constant(msh, PETSc.ScalarType(1.0e-8))
very_weak_springs_to_block_rigid_body_modes = k_springs * ufl.inner(u, u) * dx
total_energy += very_weak_springs_to_block_rigid_body_modes

# + [markdown]
# ## Solvers
# ### Displacement problem
# The displacement problem ($u$) at for fixed damage ($\alpha$) is a linear
# problem equivalent to linear elasticity with a spatially varying stiffness.
# We solve it with a standard linear solver. We use automatic differention to
# get the first derivative of the energy. We use a direct solve to solve the
# linear system, but you can also set iterative solvers and preconditioners
# when solving large problem in parallel.
# +
E_u = ufl.derivative(total_energy, u, ufl.TestFunction(V_u))
E_u_u = ufl.derivative(E_u, u, ufl.TrialFunction(V_u))
elastic_problem = SNESProblem(E_u, u, bcs_u)

b_u = la.create_petsc_vector(V_u.dofmap.index_map, V_u.dofmap.index_map_bs)
J_u = dolfinx.fem.petsc.create_matrix(elastic_problem.a)

# Create Newton solver and solve
solver_u_snes = PETSc.SNES().create()
solver_u_snes.setType("ksponly")
solver_u_snes.setFunction(elastic_problem.F, b_u)
solver_u_snes.setJacobian(elastic_problem.J, J_u)
solver_u_snes.setTolerances(rtol=1.0e-9, max_it=50)
solver_u_snes.getKSP().setType("preonly")
solver_u_snes.getKSP().setTolerances(rtol=1.0e-9)
solver_u_snes.getKSP().getPC().setType("lu")

# + [markdown]
# ### Damage problem with bound-constraint
#
# The damage problem ($\alpha$) at fixed displacement ($u$) is a variational
# inequality due to the irreversibility constraint and the bounds on the
# damage. We solve it using a specific solver for bound-constrained provided by
# PETSc, called SNESVI. To this end we define with a specific syntax a class
# defining the problem, and the lower (`lb`) and upper (`ub`) bounds.
# +
E_alpha = ufl.derivative(total_energy, alpha, ufl.TestFunction(V_alpha))
E_alpha_alpha = ufl.derivative(E_alpha, alpha, ufl.TrialFunction(V_alpha))

# + [markdown]
# We now set up the PETSc solver using petsc4py, a fully featured Python
# wrapper around PETSc.
# +
damage_problem = SNESProblem(E_alpha, alpha, bcs_alpha, J=E_alpha_alpha)

b_alpha = la.create_petsc_vector(V_alpha.dofmap.index_map, V_alpha.dofmap.index_map_bs)
J_alpha = fem.petsc.create_matrix(damage_problem.a)

# Create Newton variational inequality solver and solve
solver_alpha_snes = PETSc.SNES().create()
solver_alpha_snes.setType("vinewtonrsls")
solver_alpha_snes.setFunction(damage_problem.F, b_alpha)
solver_alpha_snes.setJacobian(damage_problem.J, J_alpha)
solver_alpha_snes.setTolerances(rtol=1.0e-9, max_it=50)
solver_alpha_snes.getKSP().setType("preonly")
solver_alpha_snes.getKSP().setTolerances(rtol=1.0e-9)
solver_alpha_snes.getKSP().getPC().setType("lu")

# Lower bound for the damage field
alpha_lb = fem.Function(V_alpha, name="lower bound")
alpha_lb.x.array[:] = 0.0
# Upper bound for the damage field
alpha_ub = fem.Function(V_alpha, name="upper bound")
alpha_ub.x.array[:] = 1.0
solver_alpha_snes.setVariableBounds(alpha_lb.vector, alpha_ub.vector)

# + [markdown]
# Before continuing we reset the displacement and damage to zero.
# +
alpha.x.array[:] = 0.0
u.x.array[:] = 0.0

# + [markdown]
# ### The static problem: solution with the alternate minimization algorithm
#
# We solve the non-linear problem in $(u,\alpha)$ at each pseudo-timestep by a
# fixed-point algorithm consisting of alternate minimization with respect to
# $u$ at fixed $\alpha$ and then for $\alpha$ at fixed $u$ until convergence is
# achieved.
#
# We now define a function that `alternate_minimization` that performs the
# alternative minimisation algorithm and assesses convergence based on the
# $L^2$ norm of the difference between the damage field at the current iterate
# and the previous iterate.
# +


def simple_monitor(u, alpha, iteration, error_L2):
    print(f"Iteration: {iteration}, Error: {error_L2:3.4e}")


def alternate_minimization(u, alpha, atol=1e-8, max_iterations=100, monitor=simple_monitor):
    alpha_old = fem.Function(alpha.function_space)
    alpha_old.x.array[:] = alpha.x.array

    for iteration in range(max_iterations):
        # Solve for displacement
        solver_u_snes.solve(None, u.vector)
        # This forward scatter is necessary when `solver_u_snes` is of type `ksponly`.
        u.x.scatter_forward()

        # Solve for damage
        solver_alpha_snes.solve(None, alpha.vector)

        # Check error and update
        L2_error = ufl.inner(alpha - alpha_old, alpha - alpha_old) * dx
        error_L2 = np.sqrt(comm.allreduce(fem.assemble_scalar(fem.form(L2_error)), op=MPI.SUM))
        alpha_old.x.array[:] = alpha.x.array

        if monitor is not None:
            monitor(u, alpha, iteration, error_L2)

        if error_L2 <= atol:
            return (error_L2, iteration)

    raise RuntimeError(
        f"Could not converge after {max_iterations} iterations, error {error_L2:3.4e}"
    )


# + [markdown]
# ## Time-stepping: solving a quasi-static problem
# +
load_c = eps_c * L  # reference value for the loading (imposed displacement)
loads = np.linspace(0, 1.5 * load_c, 20)

# Array to store results
energies = np.zeros((loads.shape[0], 3))

# File to store the results
file_name = "output/solution.xdmf"
with io.XDMFFile(comm, file_name, "w", encoding=io.XDMFFile.Encoding.HDF5) as file:
    file.write_mesh(msh)

for i_t, t in enumerate(loads):
    u_D.value = t
    energies[i_t, 0] = t

    # Update the lower bound to ensure irreversibility of damage field.
    alpha_lb.x.array[:] = alpha.x.array

    print(f"-- Solving for t = {t:3.2f} --")
    alternate_minimization(u, alpha, atol=1e-4)
    plot_damage_state(u, alpha)

    # Calculate the energies
    energies[i_t, 1] = comm.allreduce(
        dolfinx.fem.assemble_scalar(dolfinx.fem.form(elastic_energy)),
        op=MPI.SUM,
    )
    energies[i_t, 2] = comm.allreduce(
        dolfinx.fem.assemble_scalar(dolfinx.fem.form(dissipated_energy)),
        op=MPI.SUM,
    )
    with io.XDMFFile(comm, file_name, "a", encoding=io.XDMFFile.Encoding.HDF5) as file:
        file.write_function(u, t)
        file.write_function(alpha, t)

# + [markdown]
# We now plot the total, elastic and dissipated energies throughout the
# pseudo-time evolution against the applied displacement.
# +
(p3,) = plt.plot(energies[:, 0], energies[:, 1] + energies[:, 2], "ko", linewidth=2, label="Total")
(p1,) = plt.plot(energies[:, 0], energies[:, 1], "b*", linewidth=2, label="Elastic")
(p2,) = plt.plot(energies[:, 0], energies[:, 2], "r^", linewidth=2, label="Dissipated")
plt.legend()

# plt.axvline(x=eps_c * L, color="grey", linestyle="--", linewidth=2)
# plt.axhline(y=H, color="grey", linestyle="--", linewidth=2)

plt.xlabel("Displacement")
plt.ylabel("Energy")

plt.savefig("output/energies.png")

# + [markdown]
# ## Verification
#
# Let's take a look at the damage profile and verify that we acheive the
# expected solution for the AT1 model. We can easily see that the solution
# is bounded between $0$ and $1$ and that the decay to zero of the damage profile
# happens around the theoretical half-width $D$.
# +
tol = 0.001  # Avoid hitting the boundary of the mesh
num_points = 101
points = np.zeros((num_points, 3))

y = np.linspace(0.0 + tol, L - tol, num=num_points)
points[:, 0] = y
points[:, 1] = H / 2.0

fig = plt.figure()
points_on_proc, alpha_val = evaluate_on_points(alpha, points)
plt.plot(points_on_proc[:, 0], alpha_val, "k", linewidth=2, label="damage")
# plt.axvline(x=0.5 - D, color="grey", linestyle="--", linewidth=2)
# plt.axvline(x=0.5 + D, color="grey", linestyle="--", linewidth=2)
plt.grid(True)
plt.xlabel("x")
plt.ylabel(r"damage $\alpha$")
plt.legend()

# If run in parallel as a Python file, we save a plot per processor
plt.savefig(f"output/damage_line_rank_{MPI.COMM_WORLD.rank:d}.png")
