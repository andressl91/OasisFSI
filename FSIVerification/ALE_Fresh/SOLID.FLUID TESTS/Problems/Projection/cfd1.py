from dolfin import *

mesh = Mesh("Mesh/turek1.xml")
#mesh = refine(mesh)
#Parameters for each numerical case
common = {"mesh": mesh,
          "v_deg": 2,    #Velocity degree
          "p_deg": 1,    #Pressure degree
          "T": 6,          # End time
          "dt": 0.05,       # Time step
          "rho_f": 1000.,    #
          "mu_f": 1.,
          "Um": 0.2,
          "H": 0.41
     }

cfd1 = common
vars().update(cfd1)
plot(mesh, interactive=True)

for coord in mesh.coordinates():
    if coord[0]==0.6 and (0.199<=coord[1]<=0.2001): # to get the point [0.2,0.6] end of bar
        print coord
        break
# BOUNDARIES

#NOS = AutoSubDomain(lambda x: "on_boundary" and( near(x[1],0) or near(x[1], 0.41)))
Inlet = AutoSubDomain(lambda x: "on_boundary" and near(x[0],0))
Outlet = AutoSubDomain(lambda x: "on_boundary" and (near(x[0],2.5)))
Wall =  AutoSubDomain(lambda x: "on_boundary" and (near(x[1], 0.41) or near(x[1], 0)))

boundaries = FacetFunction("size_t", mesh)
boundaries.set_all(0)
DomainBoundary().mark(boundaries, 1)
Inlet.mark(boundaries, 2)
Wall.mark(boundaries, 3)
Outlet.mark(boundaries, 4)

#plot(boundaries, interactive = True)
ds = Measure("ds", subdomain_data = boundaries)
n = FacetNormal(mesh)


inlet = Expression(("1.5*Um*x[1]*(H - x[1]) / pow((H/2.0), 2)*(1 - cos(pi/2.*t))/2."\
,"0"), t = 0, Um = Um, H = H, degree = 3)

def pre_solve(t, inlet, **semimp_namespace):
    if t < 2:
        inlet.t = t
    if t <= 2:
        inlet.t = 2
    return dict(inlet = inlet)

def create_bcs(V, Q, VV, boundaries, inlet, **semimp_namespace):
    bcs_up = []
    u_geo = DirichletBC(V, Constant((0, 0)), boundaries, 1)
    u_inlet = DirichletBC(V, inlet, boundaries, 2)
    u_wall = DirichletBC(V, Constant((0, 0)), boundaries, 3)

    bcs_u = [u_geo, u_inlet, u_wall]
    #for i in bcs_u:
        #bcs_up.append(DirichletBC(VV.sub(0), i.value(), i.user_sub_domain()))
    up_geo = DirichletBC(VV.sub(0), Constant((0, 0)), boundaries, 1)
    up_inlet = DirichletBC(VV.sub(0), inlet, boundaries, 2)
    up_wall = DirichletBC(VV.sub(0), Constant((0, 0)), boundaries, 3)
    up_out = DirichletBC(VV.sub(1), Constant(0), boundaries, 4)
    bcs_up = [up_geo, up_inlet, up_wall, up_out]


    return dict(bcs_up = bcs_up, bcs_u = bcs_u)
