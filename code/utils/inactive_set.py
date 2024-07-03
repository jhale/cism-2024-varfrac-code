import numpy as np


def inactive_damage_dofs(alpha, alpha_lb, b_alpha, rtol=1e-8):
    # Find dofs where damage residual is close to 0.0 (this definition does not
    # match the one in Corrado Maurini's code)
    residual_condition = np.where(b_alpha.array > 0.0)[0]
    # Find dofs where damage is close to previous state (irreversibility/lower bound)
    near_lower_bound = np.where(np.isclose(alpha.x.array, alpha_lb.x.array, rtol=rtol))[0]
    # Find dofs where damage is close to 1.0 (upper bound)
    near_upper_bound = np.where(np.isclose(alpha.x.array, 1.0, rtol=rtol))[0]

    near_bound = np.union1d(near_upper_bound, near_lower_bound)
    active_set = np.intersect1d(near_bound, residual_condition)

    # From which we can construct the inactive set
    all_dofs = np.arange(0, alpha.function_space.dofmap.index_map.size_local, dtype=np.int32)
    inactive_dofs = np.setdiff1d(all_dofs, active_set)

    return np.sort(inactive_dofs)
