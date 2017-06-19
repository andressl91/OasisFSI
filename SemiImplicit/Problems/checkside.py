from dolfin import *
import numpy as np
import matplotlib.pyplot as plt

refi = 0
mesh_name = "base0"
mesh_file = Mesh("../Mesh/" + mesh_name +".xml")
#mesh_file = refine(mesh_file)
#Parameters for each numerical case
common = {"mesh": mesh_file}

#"checkpoint": "./FSI_fresh_checkpoints/FSI-1/P-2/dt-0.05/dvpFile.h5"
vars().update(common)
n = FacetNormal(mesh_file)
#plot(mesh, interactive=True)

for coord in mesh.coordinates():
    if coord[0]==0.6 and (0.199<=coord[1]<=0.2001): # to get the point [0.2,0.6] end of bar
        print coord
        break
# BOUNDARIES

#NOS = AutoSubDomain(lambda x: "on_boundary" and( near(x[1],0) or near(x[1], 0.41)))
Inlet = AutoSubDomain(lambda x: "on_boundary" and near(x[0],0))
Outlet = AutoSubDomain(lambda x: "on_boundary" and (near(x[0],2.5)))
Wall =  AutoSubDomain(lambda x: "on_boundary" and (near(x[1], 0.41) or near(x[1], 0)))
Bar = AutoSubDomain(lambda x: "on_boundary" and (near(x[1], 0.21)) or near(x[1], 0.19) or near(x[0], 0.6 ) )
Circle =  AutoSubDomain(lambda x: "on_boundary" and (( (x[0] - 0.2)*(x[0] - 0.2) + (x[1] - 0.2)*(x[1] - 0.2)  < 0.0505*0.0505 )  ))
Barwall =  AutoSubDomain(lambda x: "on_boundary" and (( (x[0] - 0.2)*(x[0] - 0.2) + (x[1] - 0.2)*(x[1] - 0.2)  < 0.0505*0.0505 )  and x[1]>=0.19 and x[1]<=0.21 and x[0]>0.2 ))

Bar_top = AutoSubDomain(lambda x: "on_boundary" and (near(x[1], 0.21)) )
Bar_end = AutoSubDomain(lambda x: "on_boundary" and  near(x[0], 0.6 ) )
Bar_bottom = AutoSubDomain(lambda x: "on_boundary" and (near(x[1], 0.19)) )


Bar_only = FacetFunction("size_t", mesh_file)
Bar_only.set_all(0)
Bar_top.mark(Bar_only, 1)
Bar_end.mark(Bar_only, 2)
Bar_bottom.mark(Bar_only, 3)
#plot(Bar_only, interactive=True)

dS = Measure("dS", subdomain_data = Bar_only)
print "Bar top", assemble(n("+")[0]*dS(1))
print "Bar end", assemble(n("+")[0]*dS(2))
print "Bar bottom", assemble(n("+")[0]*dS(3))


Allboundaries = DomainBoundary()

boundaries = FacetFunction("size_t",mesh_file)
boundaries.set_all(0)
Allboundaries.mark(boundaries, 1)
Wall.mark(boundaries, 2)
Inlet.mark(boundaries, 3)
Outlet.mark(boundaries, 4)
Bar.mark(boundaries, 5)
Circle.mark(boundaries, 6)
Barwall.mark(boundaries, 7)
#plot(boundaries,interactive=True)

ds = Measure("ds", subdomain_data = boundaries)
dS = Measure("dS", subdomain_data = boundaries)


Bar_area = AutoSubDomain(lambda x: (0.19 <= x[1] <= 0.21) and 0.24<= x[0] <= 0.6) # only the "flag" or "bar"
domains = CellFunction("size_t", mesh_file)
domains.set_all(1)
Bar_area.mark(domains, 2) #Overwrites structure domain
dx = Measure("dx", subdomain_data = domains)
#plot(domains,interactive = True)
dx_f = dx(1, subdomain_data = domains)
dx_s = dx(2, subdomain_data = domains)


cell_domains = CellFunction('size_t', mesh_file, 0)
solid = '&&'.join(['((0.24 - TOL < x[0]) && (x[0] < 0.6 + TOL))',
                   '((0.19 - TOL < x[1]) && (x[1] < 0.21 + TOL))'])
solid = CompiledSubDomain(solid, TOL=DOLFIN_EPS)
#Int so that solid point distance to fluid is 0
distance_f = VertexFunction('double', mesh_file, 1)
solid.mark(distance_f, 0)

#Fluid vertices
fluid_vertex_ids = np.where(distance_f.array() > 0.02)[0]

#Represent solid as its own mesh for ditance queries
solid.mark(cell_domains, 1)
solid_mesh = SubMesh(mesh_file, cell_domains, 1)
tree = solid_mesh.bounding_box_tree()

#Fill
for vertex_id in fluid_vertex_ids:
    vertex = Vertex(mesh_file, vertex_id)
    _, dist = tree.compute_closest_entity(vertex.point())
    #distance_f[vertex] = dist
    distance_f[vertex] = 1

#Build representation in a CG1 Function
P = FunctionSpace(mesh_file, 'CG', 1)
alfa = Function(P)
transform = dof_to_vertex_map(P)
data = distance_f.array()[transform]
alfa.vector().set_local(data)
alfa.vector().apply('insert')
#plot(alfa, interactive=True)

print "ALFA", assemble(alfa("-")*dS(1))
#print "Bar end", assemble(n("-")[0]*dS(2))
#print "Bar bottom", assemble(n("-")[0]*dS(3))
