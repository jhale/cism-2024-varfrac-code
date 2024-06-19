import numpy as np
from dolfinx import geometry

import basix.ufl


def evaluate_on_points(function, points):
    domain = function.function_space.mesh

    bb_tree = geometry.bb_tree(domain, domain.topology.dim)
    cells = []
    points_on_proc = []

    # Find cells whose bounding-box collide with the passed points
    cell_candidates = geometry.compute_collisions_points(bb_tree, points)

    # Choose one of the cells that contains the point
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, points)
    for i, point in enumerate(points):
        if len(colliding_cells.links(i)) > 0:
            points_on_proc.append(point)
            cells.append(colliding_cells.links(i)[0])

    points_on_proc = np.array(points_on_proc)
    if len(points_on_proc) > 0:
        values_on_proc = function.eval(points_on_proc, cells)
    else:
        values_on_proc = None
    return [points_on_proc, values_on_proc]


if __name__ == "__main__":
    from dolfinx import fem, mesh
    from mpi4py import MPI

    # Create a mesh and a first function
    domain = mesh.create_unit_square(MPI.COMM_WORLD, 8, 8, mesh.CellType.triangle)
    V_u_element = basix.ufl.element(
        "Lagrange", domain.basix_cell(), degree=1, shape=(domain.geometry.dim,)
    )
    V_u = fem.functionspace(domain, V_u_element)
    u = fem.Function(V_u)
    u.interpolate(lambda x: [1 + x[0] ** 2, 2 * x[1] ** 2])

    points = np.zeros((9, 3))
    points[:, 0] = np.linspace(0.0, 1.0, 9)
    points[:, 1] = 0.2
    points_on_proc, u_values = evaluate_on_points(u, points)
    print(u_values)
