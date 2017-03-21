from dolfin import *
"""
from Utils.argpar import *

args = parse()
v_deg = args.v_deg
p_deg = args.p_deg
d_deg = args.d_deg
dt = args.dt
"""

#Parameters for each numerical case
common = {"mesh": Mesh("Mesh/von_karman_street_FSI_fluid.xml"),
          "v_deg": 2,    #Velocity degree
          "p_deg": 1,    #Pressure degree
          "T": 8,          # End time
          "dt": 0.5,       # Time step
          "rho_f": 1000,    #
          "mu_f": 0.001,
     }

turek1 = common
vars().update(turek1)
#plot(mesh, interactive=True)
print v_deg
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

Allboundaries = DomainBoundary()

boundaries = FacetFunction("size_t",mesh)
boundaries.set_all(0)
Allboundaries.mark(boundaries, 1)
Wall.mark(boundaries, 2)
Inlet.mark(boundaries, 3)
Outlet.mark(boundaries, 4)
Bar.mark(boundaries, 5)
Circle.mark(boundaries, 6)
Barwall.mark(boundaries, 7)
#splot(boundaries,interactive=True)

ds = Measure("ds", subdomain_data = boundaries)
dS = Measure("dS", subdomain_data = boundaries)
n = FacetNormal(mesh)

Bar_area = AutoSubDomain(lambda x: (0.19 <= x[1] <= 0.21) and 0.24<= x[0] <= 0.6) # only the "flag" or "bar"
domains = CellFunction("size_t", mesh)
domains.set_all(1)
Bar_area.mark(domains, 2) #Overwrites structure domain
dx = Measure("dx", subdomain_data = domains)
#plot(domains,interactive = True)
dx_f = dx(1, subdomain_data = domains)
dx_s = dx(2, subdomain_data = domains)

#Fluid properties
#rho_f   = Constant(1.0E3)
#mu_f    = Constant(1.0)
nu = Constant(mu_f/rho_f)

Um = 0.2
D = 0.1
H = 0.41
L = 2.5
# "
inlet = Expression(("(1.5*Um*x[1]*(H - x[1]) / pow((H/2.0), 2))*(1-cos(t*pi/2.0))/2.0" \
,"0"), t = 2, Um = Um, H = H)

#Structure properties
rho_s = Constant(1.0E3)
mu_s = Constant(0.5E6)
nu_s = Constant(0.4)
E_1 = 1.4E6
lamda_s = nu_s*2*mu_s/(1 - 2*nu_s)

def create_bcs(V, Q, VV, W, boundaries, inlet, **semimp_namespace):
    bcs_up = []
    u_inlet = DirichletBC(V, inlet, boundaries, 3)
    u_wall = DirichletBC(V, Constant((0, 0)), boundaries, 2)
    p_out = DirichletBC(Q, Constant(0), boundaries, 2)
    bcs_u = [u_inlet, u_wall]
    #for i in bcs_u:
        #bcs_up.append(DirichletBC(VV.sub(0), i.value(), i.user_sub_domain()))
    up_inlet = DirichletBC(VV.sub(0), inlet, boundaries, 3)
    up_wall = DirichletBC(VV.sub(0), Constant((0, 0)), boundaries, 2)
    up_out = DirichletBC(VV.sub(1), Constant(0), boundaries, 2)
    bcs_up = [up_inlet, up_wall, up_out]


    return dict(bcs_up = bcs_up, bcs_u = bcs_u)
