import gmsh

from dolfinx.io import gmshio


def generate_mesh_with_crack(
    comm,
    Lx=1.0,
    Ly=0.5,
    Lcrack=0.3,
    lc=0.1,
    dist_min=0.1,
    dist_max=0.3,
    refinement_ratio=10,
    gdim=2,
    verbosity=4,
):
    model_rank = 0
    gmsh.initialize()

    facet_map = {"left": 1, "right": 2, "top": 3, "crack": 4, "bottom_no_crack": 5}
    cell_map = {"all": 20}

    if comm.rank == model_rank:
        model = gmsh.model()
        model.add("Rectangle")
        model.setCurrent("Rectangle")
        # Create the points
        p1 = model.geo.addPoint(0.0, 0.0, 0, lc)
        p2 = model.geo.addPoint(Lcrack, 0.0, 0, lc)
        p3 = model.geo.addPoint(Lx, 0, 0, lc)
        p4 = model.geo.addPoint(Lx, Ly, 0, lc)
        p5 = model.geo.addPoint(0, Ly, 0, lc)
        # Create the lines
        l1 = model.geo.addLine(p1, p2, tag=facet_map["crack"])
        l2 = model.geo.addLine(p2, p3, tag=facet_map["bottom_no_crack"])
        l3 = model.geo.addLine(p3, p4, tag=facet_map["right"])
        l4 = model.geo.addLine(p4, p5, tag=facet_map["top"])
        l5 = model.geo.addLine(p5, p1, tag=facet_map["left"])
        # Create the surface
        cloop1 = model.geo.addCurveLoop([l1, l2, l3, l4, l5])
        _ = model.geo.addPlaneSurface([cloop1])

        # Define the mesh size and fields for the mesh refinement
        model.mesh.field.add("Distance", 1)
        model.mesh.field.setNumbers(1, "NodesList", [p2])
        # SizeMax -                   / ------------------
        #                            /
        # SizeMin -o----------------/
        #          |                |  |
        #        Point        DistMin   DistMax
        model.mesh.field.add("Threshold", 2)
        model.mesh.field.setNumber(2, "IField", 1)
        model.mesh.field.setNumber(2, "LcMin", lc / refinement_ratio)
        model.mesh.field.setNumber(2, "LcMax", lc)
        model.mesh.field.setNumber(2, "DistMin", dist_min)
        model.mesh.field.setNumber(2, "DistMax", dist_max)
        model.mesh.field.setAsBackgroundMesh(2)
        model.geo.synchronize()

        # Assign mesh and facet tags
        surface_entities = [entity[1] for entity in model.getEntities(2)]
        model.addPhysicalGroup(2, surface_entities, tag=cell_map["all"])
        model.setPhysicalName(2, 2, "Rectangle surface")
        gmsh.option.setNumber("General.Verbosity", verbosity)
        model.mesh.generate(gdim)

        for key, value in facet_map.items():
            model.addPhysicalGroup(1, [value], tag=value)
            model.setPhysicalName(1, value, key)

        msh, cell_tags, facet_tags = gmshio.model_to_mesh(model, comm, model_rank, gdim=gdim)
        gmsh.finalize()
        msh.name = "rectangle"
        cell_tags.name = f"{msh.name}_cells"
        facet_tags.name = f"{msh.name}_facets"

        return msh, cell_tags, facet_tags, cell_map, facet_map


def generate_bar_mesh(comm, Lx, Ly, lc):
    """
    Create two-dimensional rectangle mesh using gmsh Python API.
    """
    model_rank = 0
    gmsh.initialize()
    gdim = 2
    tdim = 2

    cell_map = {"rectangle": 20}
    facet_map = {"left": 1, "right": 2, "top": 3, "bottom": 4}

    if comm.rank == model_rank:
        gmsh.option.setNumber("Mesh.Algorithm", 5)
        gmsh.model.mesh.optimize("Netgen")

        model = gmsh.model()
        model.add("rectangle")
        model.setCurrent("rectangle")
        p0 = model.geo.addPoint(0.0, 0.0, 0.0, lc)
        p1 = model.geo.addPoint(Lx, 0.0, 0.0, lc)
        p2 = model.geo.addPoint(Lx, Ly, 0.0, lc)
        p3 = model.geo.addPoint(0, Ly, 0.0, lc)
        bottom = model.geo.addLine(p0, p1, tag=facet_map["bottom"])
        right = model.geo.addLine(p1, p2, tag=facet_map["right"])
        top = model.geo.addLine(p2, p3, tag=facet_map["top"])
        left = model.geo.addLine(p3, p0, tag=facet_map["left"])
        cloop1 = model.geo.addCurveLoop([bottom, right, top, left])
        model.geo.addPlaneSurface([cloop1])

        model.geo.synchronize()
        surface_entities = [model[1] for model in model.getEntities(tdim)]
        model.addPhysicalGroup(tdim, surface_entities, tag=0)
        model.setPhysicalName(tdim, 0, "rectangle")

        # Set geometric order of mesh cells
        gmsh.model.mesh.setOrder(1)
        for key, value in facet_map.items():
            model.addPhysicalGroup(tdim - 1, [value], tag=value)
            model.setPhysicalName(tdim - 1, value, key)

        model.mesh.generate(tdim)

        msh, cell_tags, facet_tags = gmshio.model_to_mesh(model, comm, model_rank, gdim=gdim)
        gmsh.finalize()

        msh.name = "rectangle"
        cell_tags.name = f"{msh.name}_cells"
        facet_tags.name = f"{msh.name}_facets"

        return msh, cell_tags, facet_tags, cell_map, facet_map
