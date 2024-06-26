# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
# ---

# + [markdown]

import sys

from petsc4py import PETSc

import numpy as np
from dolfiny.restriction import Restriction

import basix
import basix.ufl
import dolfinx
import dolfinx.fem as fem
import dolfinx.fem.petsc
import dolfinx.la as la
import dolfinx.plot as plot
import ufl

sys.path.append("../utils/")

from meshes import generate_bar_mesh
from petsc_problems import SNESProblem

# By dimensional analysis
sigma_c = 1.0  #
Lx = 1.0  # Size of domain in x-direction
Ly = 0.1  # Size of domain in y-direction

# Define free parameters. Table 1 Zelosi and Maurini, first row.
G_c = 1.0  # Fracture toughness.
E_0 = 1.0  # Young's modulus.
ell = 0.05  # Regularisation length scale.

# Additional parameters
nu_0 = 0.3  # Poisson's ratio.

# Computational parameters
pre_damage_num_steps = 10  # Number of load steps before damage
post_damage_num_steps = 50  # Number of load steps after damage
lc = ell / 5.0  # Characteristic mesh size

# Derived quantities
mu_0 = E_0 / (2 * (1 + nu_0))
kappa_0 = E_0 / (2 * (1 - nu_0))
w_1 = G_c / (np.pi * ell)
gamma_traction = (2 * w_1 * E_0) / (sigma_c**2)
t_c = sigma_c / E_0
t_f = gamma_traction * t_c
t_star = 2 * np.pi * ell / Lx * w_1 / sigma_c

loads = np.linspace(0.0, t_c, pre_damage_num_steps)
msh, mt, ft, mm, fm = generate_bar_mesh(Lx=Lx, Ly=Ly, lc=lc)

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
alpha_ub.x.array[:] = 1.0

dx = ufl.Measure("dx", domain=msh)
ds = ufl.Measure("ds", domain=msh, subdomain_data=ft)

dofs_alpha_left = dolfinx.fem.locate_dofs_topological(
    V_alpha, msh.topology.dim - 1, ft.find(fm["left"])
)

dofs_alpha_right = dolfinx.fem.locate_dofs_topological(
    V_alpha, msh.topology.dim - 1, ft.find(fm["right"])
)

dofs_ux_left = dolfinx.fem.locate_dofs_topological(
    V_u.sub(0), msh.topology.dim - 1, ft.find(fm["left"])
)

dofs_ux_right = dolfinx.fem.locate_dofs_topological(
    V_u.sub(0), msh.topology.dim - 1, ft.find(fm["right"])
)

dofs_uy_left = dolfinx.fem.locate_dofs_topological(
    V_u.sub(1), msh.topology.dim - 1, ft.find(fm["left"])
)

ux_right = fem.Constant(msh, 0.0)
bcs_u = [
    fem.dirichletbc(fem.Constant(msh, 0.0), dofs_ux_left, V_u.sub(0)),
    fem.dirichletbc(fem.Constant(msh, 0.0), dofs_uy_left, V_u.sub(1)),
    fem.dirichletbc(ux_right, dofs_ux_right, V_u.sub(0)),
]

bcs_alpha = [
    fem.dirichletbc(fem.Constant(msh, 0.0), dofs_alpha_left, V_alpha),
    fem.dirichletbc(fem.Constant(msh, 0.0), dofs_alpha_right, V_alpha),
]

bcs_all = bcs_u + bcs_alpha

# Set boundary condition on damage upper bound
fem.set_bc(alpha_ub.x.array, bcs_alpha)
alpha_ub.x.scatter_forward()


def eps(u):
    return ufl.sym(ufl.grad(u))


def w(alpha):
    return 1.0 - (1.0 - alpha) ** 2


def a(alpha, gamma=gamma_traction, kres=1.0e-8):
    return (1.0 - w(alpha)) / (1.0 + w(alpha) * (gamma - 1.0)) + kres


def mu(alpha, gamma=gamma_traction):
    return dolfinx.fem.Constant(msh, mu_0) * a(alpha, gamma=fem.Constant(msh, gamma_traction))


def kappa(alpha, gamma=gamma_traction):
    return dolfinx.fem.Constant(msh, kappa_0) * a(
        alpha, gamma=dolfinx.fem.Constant(msh, gamma_traction)
    )


def damage_dissipation_density(alpha):
    grad_alpha = ufl.grad(alpha)
    w_1_ = dolfinx.fem.Constant(msh, w_1)
    ell_2 = dolfinx.fem.Constant(msh, ell * ell)
    return w_1_ * (w(alpha) + ell_2 * ufl.inner(grad_alpha, grad_alpha))


def elastic_deviatoric_energy_density(eps, alpha):
    return mu(alpha) * ufl.inner(ufl.dev(eps), ufl.dev(eps))


def elastic_isotropic_energy_density(eps, alpha):
    return 0.5 * kappa(alpha) * ufl.tr(eps) * ufl.tr(eps)


def elastic_energy_density(eps, alpha):
    return elastic_deviatoric_energy_density(eps, alpha) + elastic_isotropic_energy_density(
        eps, alpha
    )


def total_energy(u, alpha):
    return elastic_energy_density(eps(u), alpha) * dx + damage_dissipation_density(alpha) * dx


def sigma(eps, alpha):
    eps_ = ufl.variable(eps)
    sigma = ufl.diff(elastic_energy_density(eps_, alpha), eps_)
    return sigma


energy = total_energy(u, alpha)

# Overall algorithm:
# 1. Continuation in displacement (outer loop).
# 2. Find minima using alternate minimisation.
# 3. Assess stability of reduced fully coupled problem.
# a. Assemble full Jacobian and residual.
# b. Identify active set.
# c. Remove degrees of freedom in the active set from the damage problem.
# d. Find eigenvalues of reduced block problem (dolfiny?).

# Move this all out to function.
E_u = ufl.derivative(energy, u, ufl.TestFunction(V_u))
E_u_u = ufl.derivative(E_u, u, ufl.TrialFunction(V_u))
elastic_problem = SNESProblem(E_u, u, bcs_u, J=E_u_u)

b_u = la.create_petsc_vector(V_u.dofmap.index_map, V_u.dofmap.index_map_bs)
J_u = dolfinx.fem.petsc.create_matrix(elastic_problem.a)

b_u = la.create_petsc_vector(V_u.dofmap.index_map, V_u.dofmap.index_map_bs)
J_u = dolfinx.fem.petsc.create_matrix(elastic_problem.a)


# Setup linear elasticity problem and solve
solver_u_snes = PETSc.SNES().create()
solver_u_snes.setType("ksponly")
solver_u_snes.setFunction(elastic_problem.F, b_u)
solver_u_snes.setJacobian(elastic_problem.J, J_u)
solver_u_snes.setTolerances(rtol=1.0e-8, max_it=50)
solver_u_snes.getKSP().setType("preonly")
solver_u_snes.getKSP().setTolerances(rtol=1.0e-8)
solver_u_snes.getKSP().getPC().setType("lu")

# Setup damage problem.
E_alpha = ufl.derivative(energy, alpha, ufl.TestFunction(V_alpha))
E_alpha_alpha = ufl.derivative(E_alpha, alpha, ufl.TrialFunction(V_alpha))

damage_problem = SNESProblem(E_alpha, alpha, bcs_alpha, J=E_alpha_alpha)

b_alpha = la.create_petsc_vector(V_alpha.dofmap.index_map, V_alpha.dofmap.index_map_bs)
J_alpha = fem.petsc.create_matrix(damage_problem.a)

# Create Newton variational inequality solver and solve
solver_alpha_snes = PETSc.SNES().create()
solver_alpha_snes.setType("vinewtonrsls")
solver_alpha_snes.setFunction(damage_problem.F, b_alpha)
solver_alpha_snes.setJacobian(damage_problem.J, J_alpha)
solver_alpha_snes.setTolerances(rtol=1.0e-8, max_it=50)
solver_alpha_snes.getKSP().setType("preonly")
solver_alpha_snes.getKSP().getPC().setType("lu")
solver_alpha_snes.setVariableBounds(alpha_lb.vector, alpha_ub.vector)


def simple_monitor(u, alpha, iteration, error_L2):
    print(f"Iteration: {iteration}, Error: {error_L2:3.4e}")


def alternate_minimization(u, alpha, atol=1e-6, max_iter=100, monitor=simple_monitor):
    alpha_old = fem.Function(alpha.function_space)
    alpha_old.x.array[:] = alpha.x.array

    for iteration in range(max_iter):
        # Solve displacement
        solver_u_snes.solve(None, u.vector)

        # Solve damage
        solver_alpha_snes.solve(None, alpha.vector)

        # check error and update
        L2_error = ufl.inner(alpha - alpha_old, alpha - alpha_old) * dx
        error_L2 = np.sqrt(fem.assemble_scalar(fem.form(L2_error)))
        alpha.vector.copy(alpha_old.vector)

        if monitor is not None:
            monitor(u, alpha, iteration, error_L2)

        if error_L2 <= atol:
            return (error_L2, iteration)

    raise RuntimeError(f"Could not converge after {max_iter} iterations, error {error_L2:3.4e}")


# Define the fully coupled block problem for stability analysis
element_lmbda = basix.ufl.element("Lagrange", msh.basix_cell(), degree=0, discontinuous=True)
V_lmbda = fem.functionspace(msh, element_lmbda)
lmbda = fem.Function(V_lmbda)

# Block residual
F = [None] * 2
F[0] = ufl.derivative(energy + lmbda * ufl.inner(u, u) * dx, u, ufl.TestFunction(V_u))
F[1] = ufl.derivative(
    energy + lmbda * ufl.inner(alpha, alpha) * dx, alpha, ufl.TestFunction(V_alpha)
)

# Block M
M = [[None] * 2] * 2
M[0][0] = ufl.derivative(F[0], u, ufl.TrialFunction(V_u))
M[0][1] = ufl.derivative(F[0], alpha, ufl.TrialFunction(V_alpha))
M[1][0] = ufl.derivative(F[1], u, ufl.TrialFunction(V_u))
M[1][1] = ufl.derivative(F[1], alpha, ufl.TrialFunction(V_alpha))

# Block A
A = [[None] * 2] * 2
for i in range(2):
    for j in range(2):
        A[i][j] = ufl.replace(M[i][j], {lmbda: ufl.zero()})

# Block B
B = [[None] * 2] * 2
for i in range(2):
    for j in range(2):
        B[i][j] = ufl.algorithms.expand_derivatives(ufl.diff(M[i][j], lmbda))
        B[i][j] = ufl.replace(B[i][j], {lmbda: ufl.zero()})

        if B[i][j].empty():
            B[i][j] = None

A_form = fem.form(A)
B_form = fem.form(B)

A = fem.petsc.create_matrix_block(A_form)
B = fem.petsc.create_matrix_block(B_form)

for i_t, t in enumerate(loads):
    ux_right.value = t * t_c

    print(f"-- Solving for t = {t:3.2f} --")

    # Update the lower bound to ensure irreversibility of damage field.
    alpha.vector.copy(alpha_lb.vector)
    alpha_lb.x.scatter_forward()
    alternate_minimization(u, alpha)

    # Assemble operators on union of active (damaged) and inactive (undamaged)
    # sets.
    A.zeroEntries()
    fem.petsc.assemble_matrix_block(A, A_form, bcs=bcs_all)
    A.assemble()

    B.zeroEntries()
    fem.petsc.assemble_matrix_block(B, B_form, bcs=bcs_all)
    B.assemble()

    # Get inactive sets.
    u_inactive_set = np.arange(0, V_u.dofmap.index_map.size_local, dtype=np.int32)
    # Get inactive sets.
    alpha_inactive_set = solver_alpha_snes.getVIInactiveSet()

    restriction = Restriction([V_u, V_alpha], [u_inactive_set, alpha_inactive_set])

    # Check stability using reduced system using SLEPc.
