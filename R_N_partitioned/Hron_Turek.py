from dolfin import *
import matplotlib.pyplot as plt
import numpy as np
import time
import sys
from parser import *
#from FSI_ALE_Partitioned import *

mesh = Mesh("fluid_new.xml")
h = mesh.hmin()
for coord in mesh.coordinates():
    if coord[0]==0.6 and (0.199<=coord[1]<=0.2001): # to get the point [0.2,0.6] end of bar
        print coord
        break

V1 = VectorFunctionSpace(mesh, "CG", 1) # Fluid velocity
V2 = VectorFunctionSpace(mesh, "CG", 1) # displacement
Q  = FunctionSpace(mesh, "CG", 1)       # Fluid Pressure

VQ = MixedFunctionSpace([V1,Q])

#NOS = AutoSubDomain(lambda x: "on_boundary" and( near(x[1],0) or near(x[1], 0.41)))
Inlet = AutoSubDomain(lambda x: "on_boundary" and near(x[0],0))
Outlet = AutoSubDomain(lambda x: "on_boundary" and (near(x[0],2.5)))
Wall =  AutoSubDomain(lambda x: "on_boundary" and (near(x[1], 0.41) or near(x[1], 0)))
Bar = AutoSubDomain(lambda x: "on_boundary" and (near(x[1], 0.21)) or near(x[1], 0.19) or near(x[0], 0.6 ) )
Circle =  AutoSubDomain(lambda x: "on_boundary" and (( (x[0] - 0.2)*(x[0] - 0.2) + (x[1] - 0.2)*(x[1] - 0.2)  < 0.0505*0.0505 )  ))
Barwall =  AutoSubDomain(lambda x: "on_boundary" and (( (x[0] - 0.2)*(x[0] - 0.2) + (x[1] - 0.2)*(x[1] - 0.2)  < 0.0505*0.0505 )  and x[1]>=0.19 and x[1]<=0.21 and x[0]>0.2 ))

Allboundaries = DomainBoundary()

Bar_area = AutoSubDomain(lambda x: (0.19 <= x[1] <= 0.21) and 0.24<= x[0] <= 0.6) # only the "flag" or "bar"


boundaries = FacetFunction("size_t",mesh)
boundaries.set_all(0)
Allboundaries.mark(boundaries, 1)
Wall.mark(boundaries, 2)
Inlet.mark(boundaries, 3)
Outlet.mark(boundaries, 4)
Bar.mark(boundaries, 5)
Circle.mark(boundaries, 6)
Barwall.mark(boundaries, 7)
#Bar_area.mark(boundaries, 8)
#plot(boundaries,interactive=True)


ds = Measure("ds", subdomain_data = boundaries)
dS = Measure("dS", subdomain_data = boundaries)
n = FacetNormal(mesh)

domains = CellFunction("size_t",mesh)
domains.set_all(1)
Bar_area.mark(domains,2) #Overwrites structure domain
dx = Measure("dx",subdomain_data=domains)
#plot(domains,interactive = True)
dx_f = dx(1,subdomain_data=domains)
dx_s = dx(2,subdomain_data=domains)

#displacement conditions:
d_wall    = DirichletBC(V1, ((0.0, 0.0)), boundaries, 2)
d_inlet   = DirichletBC(V1, ((0.0, 0.0)), boundaries, 3)
d_outlet  = DirichletBC(V1, ((0.0, 0.0)), boundaries, 4)
d_circle  = DirichletBC(V1, ((0.0, 0.0)), boundaries, 6)
d_barwall = DirichletBC(V1, ((0.0, 0.0)), boundaries, 7) #No slip on geometry in fluid

#Pressure Conditions
p_out = DirichletBC(VQ.sub(1), 0, boundaries, 4)
# FLUID
FSI = FSI_deg
nu = 10**-3
rho_f = 1.0*1e3
mu_f = rho_f*nu
U_in = [0.2, 1.0, 2.0][FSI-1]   # Reynolds vel

# SOLID
Pr = 0.4
mu_s = [0.5, 0.5, 2.0][FSI-1]*1e6
rho_s = [1.0, 10, 1.0][FSI-1]*1e3
lamda_s = 2*mu_s*Pr/(1-2.*Pr)
H = 0.41
L = 2.5


class inlet(Expression):
    #def __init__(self):
    #    self.t = 0
    def init(self):
        self.t = 0

    def set_t(self, t):
        self.t = t

    def eval(self,value,x):
        value[0] = 0.5*(1-np.cos(self.t*np.pi/2))*1.5*U_in*x[1]*(H-x[1])/((H/2.0)**2)
        value[1] = 0

    def value_shape(self):
        return (2,)

inlet = inlet()
inlet.init()
#Fluid velocity conditions
u_inlet  = DirichletBC(VQ.sub(0), inlet, boundaries, 3)
u_wall   = DirichletBC(VQ.sub(0), ((0.0, 0.0)), boundaries, 2)
u_circ   = DirichletBC(VQ.sub(0), ((0.0, 0.0)), boundaries, 6) #No slip on geometry in fluid
u_barwall= DirichletBC(VQ.sub(0), ((0.0, 0.0)), boundaries, 7) #No slip on geometry in fluid
u_bar= DirichletBC(VQ.sub(0), ((0.0, 0.0)), boundaries, 5) #No slip on geometry in fluid

#Fluid velocity conditions

bc_u = [u_inlet, u_wall, u_circ, u_barwall]#,u_bar]
#bc_d = [d_wall, d_inlet, d_outlet, d_circle, d_barwall] # TODO: Only need barwall? Ident zeros
bc_d = [d_barwall] # TODO: Only need barwall? Ident zeros
bc_p = [p_out]
#bcs = bc_u + bc_d + bc_p
