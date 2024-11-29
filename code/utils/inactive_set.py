import numpy as np


def inactive_damage_dofs(alpha, alpha_lb, b_alpha, bc_dofs_alpha, rtol=1e-8):
    """
    Calculate the inactive set for the damage sub-problem.

    Args:
        alpha: The current damage.
        alpha_lb: The previous lower bound on the damage.
        b_alpha: The current residual.
        bc_dofs_alpha: Degrees of freedom on which damage is applied
        rtol: The relative tolerance for determining closeness to the lower and
              upper bound.
    """
    # Find dofs where magnitude of damage residual is close to 0.0 (this
    # definition does not match the one in Corrado Maurini's code)
    # Find dofs where damage is close to previous state (irreversibility/lower bound)
    near_lower_bound = np.where(np.isclose(alpha.x.array, alpha_lb.x.array, rtol=rtol))[0]
    # Find dofs where damage is close to 1.0 (upper bound)
    near_upper_bound = np.where(np.isclose(alpha.x.array, 1.0, rtol=rtol))[0]
    near_bounds = np.union1d(near_upper_bound, near_lower_bound)

    residual_condition = np.where(np.abs(b_alpha.array) > 0.0)[0]
    active = np.union1d(np.intersect1d(near_bounds, residual_condition), bc_dofs_alpha)

    # From which we can construct the inactive set
    all = np.arange(0, alpha.function_space.dofmap.index_map.size_local, dtype=np.int32)
    inactive = np.setdiff1d(all, active)

    return np.sort(inactive)
