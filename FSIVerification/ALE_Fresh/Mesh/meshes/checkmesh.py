from dolfin import *

mesh1 = Mesh("./base0.xml")
mesh2 = Mesh("./base1.xml")
mesh3 = Mesh("./base2.xml")
mesh4 = Mesh("./fluid_new.xml")
mesh5 = refine(mesh4)
for i in [mesh1, mesh2, mesh3, mesh4, mesh5]:
    V = VectorFunctionSpace(i, "CG", 2)
    P = FunctionSpace(i, "CG", 1)
    W = V*P
    i.init()
    tet_inds = [cell.index() for cell in cells(i)]
    facet_inds = [facet.index() for facet in facets(i)]
    point_inds = [vert.index() for vert in vertices(i)]
    print "New mesh"
    print "Dofs", W.dim()
    print "Number of Cells, Elements", i.num_cells()
    print "#################"
"""
omega = UnitSquareMesh(2,2)
omega.init()
tet_inds = [cell.index() for cell in cells(omega)]
facet_inds = [facet.index() for facet in facets(omega)]
point_inds = [vert.index() for vert in vertices(omega)]
print tet_inds
print facet_inds
print point_inds

plot(omega, interactive=True)
"""
