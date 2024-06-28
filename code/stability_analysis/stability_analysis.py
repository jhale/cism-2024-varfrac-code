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
# # Stability analysis of a selective linear-softening gradient damage model
#
# *Authors:*
# - Jack S. Hale (University of Luxembourg)
# - Corrado Maurini (Sorbonne Université)
#
# In this notebook we implement a stability analysis on a selective
# linear-softening (S-LS) gradient damage model. For a full overview of the
# theoretical aspects we refer the reader to:
#
# - Stability and crack nucleation in variational phase-field models of
# fracture: effects of length-scales and stress multi-axiality. Zolesi and
# Maurini 2024. Preprint. https://hal.sorbonne-universite.fr/hal-04552309
#
# It is well understood that the classical phase-field model of brittle
# fracture, e.g. AT1 or AT2 type models, converge asymptotically to the
# Griffith fracture model as the regularisation length goes to zero. This
# result guarantees that *pre-existing* cracks propagate consistently with the
# Griffith model.
#
# It is also widely observed that phase-field models allow for the nucleation of new
# cracks from a completely undamaged state (i.e. without *pre-existing* cracks).
#
# The essence of the approach is as follows:
#
# 1. We define a S-LS gradient damage model via its energy functional
#    $\mathcal{E}$ with a split of the elastic energy into deviatoric and
#    spherical parts, with each associated with a selective softening parameter.
# 2. Following the previous tutorial, we solve for the problem state $(u_t,
#    \alpha_t)$ in pseudo-time $t$ using the alternate minimisation algorithm with a
#    pointwise bound constraint to ensure irreversibility $\dot{\alpha}_t \ge 0$.
# 3. To understand the stability of the equilibrium state we then solve an
#    eigenvalue problem associated with the second-order state stability
#    condition. An equilibrium state $(u_t, \alpha_t)$ is said to be stable if the
#    second derivative (Hessian) of the energy on the active set (all variables
#    except those where the damage constraint is active) is positive:
#    $$
#    \mathcal{E}^{''}(u, \alpha)(v, \beta) > 0, \quad \forall (v, \beta)
#    \in \mathcal{C}_0 \times \mathcal{D}^{+}_0,
#    $$
#    where $\mathcal{C}_0$ is the space of displacements with vanishing value
#    on the boundary and $\mathcal{D}_0^+$ the damage field with vanishing value
#    on the boundary and with damage $\alpha \ge 0$.
#
# The numerical aspects of solving this problem will be discussed below.
#
# ## Preamble
#
# We begin by importing the required Python modules.
#
# Here we use will use the restriction functionality of
# [dolfiny](https://github.com/michalhabera/dolfiny).
#
# You can install dolfiny in your container by opening a shell
# and running:
#
#    pip install git+https://github.com/michalhabera/dolfiny.git@v0.8.0
#
# + 
import sys

from mpi4py import MPI
from petsc4py import PETSc

import numpy as np
from dolfiny.restriction import Restriction
from slepc4py import SLEPc

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

# + [markdown]
# ## Mesh
#
# We define the mesh using a provided function which uses gmsh internally. This
# function returns the mesh and so-called mesh tags `mt` and facet tags `ft`
# that allow us to integrate over, or apply boundary conditions to, specific
# parts of the boundary. The dictionaries `mm` and `fm` contain a map between
# easy-to-remember strings and numbers in the tags `mt` and `ft`, respectively.
# +
Lx = 1.0  # Size of domain in x-direction
Ly = 0.1  # Size of domain in y-direction

# Define free parameters for SL-S model. Table 1 Zelosi and Maurini, third row.
E_0 = 1.0  # Young's modulus.
G_c = 1.0  # Fracture toughness.
sigma_peak = 1.0  # Peak strength.
ell = 0.05  # Regularisation length scale.

comm = MPI.COMM_WORLD
msh, mt, ft, mm, fm = generate_bar_mesh(comm, Lx=Lx, Ly=Ly, lc=ell / 5.0)

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

u = fem.Function(V_u, name="displacement")
alpha = fem.Function(V_alpha, name="damage")

alpha_lb = dolfinx.fem.Function(V_alpha, name="lower bound")
alpha_ub = dolfinx.fem.Function(V_alpha, name="upper bound")
alpha_ub.x.array[:] = 1.0

dx = ufl.Measure("dx", domain=msh)
ds = ufl.Measure("ds", domain=msh)

# + [markdown]
# ### Boundary conditions
# We impose Dirichlet boundary conditions on the displacement and the damage
# field on the appropriate parts of the boundary.
#
# In the previous example we use predicates to locate the facets on the
# boundary. An alternative approach is to directly use the facet tags `fm`
# returned directly by the mesh generator. This can be helpful when dealing
# with applying boundary conditions on meshes with complex curved surfaces.
# +
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

dofs_alpha_left = dolfinx.fem.locate_dofs_topological(
    V_alpha, msh.topology.dim - 1, ft.find(fm["left"])
)

dofs_alpha_right = dolfinx.fem.locate_dofs_topological(
    V_alpha, msh.topology.dim - 1, ft.find(fm["right"])
)

bcs_alpha = [
    fem.dirichletbc(fem.Constant(msh, 0.0), dofs_alpha_left, V_alpha),
    fem.dirichletbc(fem.Constant(msh, 0.0), dofs_alpha_right, V_alpha),
]

bcs_all = bcs_u + bcs_alpha

# Set boundary condition on damage upper bound
fem.set_bc(alpha_ub.x.array, bcs_alpha)
alpha_ub.x.scatter_forward()

# + [markdown]
# ## Variational formulation of the problem
# ### Constitutive model
#
# We will now define the L-SL constitutive model and the related parameters. In
# turn these will be used to define the energy.
#
# The strain energy density of the S-LS constitutive model can be written as
#
# $$
# \mathcal{E}(\varepsilon, \alpha) = \frac{\kappa(\alpha)}{2} [\mathrm{tr}(\varepsilon)]^2 +
# \mu(\alpha) | \mathrm{dev}(\varpepsilon) |^2 + w_1(w(\alpha) + \ell^2 | \grad
# \alpha | ^ 2),
# $$
#
# where $\varepsilon$ is the usual small strain tensor as a function of the
# displacements $u$, $\mathrm{tr} is the trace operator, $\mathrm{dev}$ is the
# deviatoric operator, and $alpha$ is the damage field. The dissipation
# function is
#
# $$
# w(\alpha) := 1 - (1 - \alpha^2),
# $$
#
# and spherical and deviatoric degradation functions are
#
# $$
# \kappa(\alpha) := \frac{1 - w(\alpha)}{1 + (\gamma_{\kappa} - 1) w (\alpha)} \kappa_0,
# $$
#
# $$
# \mu(\alpha) := \frac{1 - w(\alpha)}{1 + (\gamma_{\mu} - 1) w (\alpha)} \mu_0.
# $$
# +

# Additional parameters
nu_0 = 0.3  # Poisson's ratio.

# Derived quantities from four free parameters
mu_0 = E_0 / (2.0 * (1.0 + nu_0))  # First Lame parameter of undamaged material
kappa_0 = E_0 / (2.0 * (1 - nu_0))  # Bulk modulus of undamaged material
w_1 = G_c / (np.pi * ell)  # Energy required to damage unit volume in a homogeneous process

# Softening parameters for spherical and deviatoric contributions
gamma_mu = gamma_kappa = (2 * G_c * E_0) / (np.pi * ell * sigma_peak**2)
t_peak = sigma_peak / E_0  # Peak traction
t_star = 2 * G_c / (sigma_peak * Lx)  # Final failure traction for homogeneous solution
t_f = 2 * G_c / (sigma_peak * np.pi * ell)  # Final failure traction for localised solution

# Computational parameters
pre_damage_num_steps = 10  # Number of load steps before damage
post_damage_num_steps = 50  # Number of load steps after damage

loads = np.linspace(0.0, t_peak, pre_damage_num_steps)


def eps(u):
    return ufl.sym(ufl.grad(u))


def w(alpha):
    return 1.0 - (1.0 - alpha) ** 2


def mu(alpha):
    return mu_0 * (1.0 - w(alpha)) / (1.0 + (gamma_mu - 1.0) * w(alpha))


def kappa(alpha):
    return kappa_0 * (1.0 - w(alpha)) / (1.0 + (gamma_kappa - 1.0) * w(alpha))


def damage_dissipation_density(alpha):
    w_1_ = dolfinx.fem.Constant(msh, w_1)
    ell_squared = dolfinx.fem.Constant(msh, ell * ell)
    return w_1_ * (w(alpha) + ell_squared * ufl.inner(ufl.grad(alpha), ufl.grad(alpha)))


def deviatoric_energy_density(eps, alpha):
    return mu(alpha) * ufl.inner(ufl.dev(eps), ufl.dev(eps))


def isotropic_energy_density(eps, alpha):
    return 0.5 * kappa(alpha) * ufl.tr(eps) * ufl.tr(eps)


def elastic_energy_density(eps, alpha):
    return deviatoric_energy_density(eps, alpha) + isotropic_energy_density(
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
solver_u_snes.setOptionsPrefix("elasticity_")
solver_u_snes.setFunction(elastic_problem.F, b_u)
solver_u_snes.setJacobian(elastic_problem.J, J_u)

opts = PETSc.Options()
opts["elasticity_snes_type"] = "ksponly"
opts["elasticity_ksp_type"] = "preonly"
opts["elasticity_pc_type"] = "lu"
opts["elasticity_pc_factor_mat_solver_type"] = "mumps"
opts["elasticity_snes_rtol"] = 1.0e-8
opts["elasticity_snes_atol"] = 1.0e-8
solver_u_snes.setFromOptions()

# Setup damage problem.
E_alpha = ufl.derivative(energy, alpha, ufl.TestFunction(V_alpha))
E_alpha_alpha = ufl.derivative(E_alpha, alpha, ufl.TrialFunction(V_alpha))

damage_problem = SNESProblem(E_alpha, alpha, bcs_alpha, J=E_alpha_alpha)

b_alpha = la.create_petsc_vector(V_alpha.dofmap.index_map, V_alpha.dofmap.index_map_bs)
J_alpha = fem.petsc.create_matrix(damage_problem.a)

# Create Newton variational inequality solver and solve
solver_alpha_snes = PETSc.SNES().create()
solver_alpha_snes.setOptionsPrefix("damage_")
solver_alpha_snes.setFunction(damage_problem.F, b_alpha)
solver_alpha_snes.setJacobian(damage_problem.J, J_alpha)
solver_alpha_snes.setVariableBounds(alpha_lb.vector, alpha_ub.vector)

opts["damage_snes_type"] = "vinewtonrsls"
opts["damage_ksp_type"] = "preonly"
opts["damage_pc_type"] = "lu"
opts["damage_pc_factor_mat_solver_type"] = "mumps"
opts["damage_snes_rtol"] = 1.0e-8
opts["damage_snes_atol"] = 1.0e-8
opts["damage_snes_max_it"] = 50

solver_alpha_snes.setFromOptions()


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
# Block residual
F = [None for i in range(2)]
F[0] = ufl.derivative(energy, u, ufl.TestFunction(V_u))
F[1] = ufl.derivative(energy, alpha, ufl.TestFunction(V_alpha))

# Block A
A = [[None for i in range(2)] for j in range(2)]
A[0][0] = ufl.derivative(F[0], u, ufl.TrialFunction(V_u))
A[0][1] = ufl.derivative(F[0], alpha, ufl.TrialFunction(V_alpha))
A[1][0] = ufl.derivative(F[1], u, ufl.TrialFunction(V_u))
A[1][1] = ufl.derivative(F[1], alpha, ufl.TrialFunction(V_alpha))

# Block B
B = [[None for i in range(2)] for j in range(2)]
B[0][0] = ufl.inner(ufl.TrialFunction(V_u), ufl.TestFunction(V_u)) * dx
B[1][1] = ufl.inner(ufl.TrialFunction(V_alpha), ufl.TestFunction(V_alpha)) * dx

A_form = fem.form(A)
B_form = fem.form(B)

A = fem.petsc.create_matrix_block(A_form)
B = fem.petsc.create_matrix_block(B_form)

# SLEPc solver
stability_solver = SLEPc.EPS().create()
stability_solver.setOptionsPrefix("stability_")

opts["stability_eps_type"] = "krylovschur"
opts["stability_eps_target"] = "smallest_real"
opts["stability_eps_target"] = 1e-5
opts["stability_st_type"] = "sinvert"
opts["stability_st_shift"] = -0.1
opts["stability_eps_tol"] = 1e-7
opts["stability_st_ksp_type"] = "preonly"
opts["stability_st_pc_type"] = "cholesky"
opts["stability_st_pc_factor_mat_solver_type"] = "mumps"
opts["stability_st_mat_mumps_icntl_24"] = 1
stability_solver.setFromOptions()

for i_t, t in enumerate(loads):
    ux_right.value = t * t_peak

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
    alpha_inactive_set = solver_alpha_snes.getVIInactiveSet().array

    restriction = Restriction([V_u, V_alpha], [u_inactive_set, alpha_inactive_set])

    # Create restricted operators.
    A_restricted = restriction.restrict_matrix(A)
    B_restricted = restriction.restrict_matrix(B)

    stability_solver.setOperators(A_restricted, B_restricted)
    stability_solver.solve()

    num_converged = stability_solver.getConverged()

    print(num_converged)

    for i in range(0, num_converged):
        print(stability_solver.getEigenvalue(i))
