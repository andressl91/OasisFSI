from dolfin import *
from mapping import *
import numpy as np

def Hron_Turek_bcs(VVQ,mesh,U_in,H):
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

    class inlet(Expression):
    	def __init__(self):
    		self.t = 0
    	def eval(self,value,x):
    		value[0] = 0.5*(1-np.cos(self.t*np.pi/2))*1.5*U_in*x[1]*(H-x[1])/((H/2.0)**2)
    		value[1] = 0
    	def value_shape(self):
    		return (2,)

    inlet = inlet()
    #Fluid velocity conditions
    u_inlet  = DirichletBC(VVQ.sub(0), inlet, boundaries, 3)
    u_wall   = DirichletBC(VVQ.sub(0), ((0.0, 0.0)), boundaries, 2)
    u_circ   = DirichletBC(VVQ.sub(0), ((0.0, 0.0)), boundaries, 6) #No slip on geometry in fluid
    u_bar    = DirichletBC(VVQ.sub(0), ((0.0, 0.0)), boundaries, 5) #No slip on geometry in fluid
    u_barwall= DirichletBC(VVQ.sub(0), ((0.0, 0.0)), boundaries, 7) #No slip on geometry in fluid

    #displacement conditions:
    d_wall    = DirichletBC(VVQ.sub(1), ((0.0, 0.0)), boundaries, 2)
    d_inlet   = DirichletBC(VVQ.sub(1), ((0.0, 0.0)), boundaries, 3)
    d_outlet  = DirichletBC(VVQ.sub(1), ((0.0, 0.0)), boundaries, 4)
    d_circle  = DirichletBC(VVQ.sub(1), ((0.0, 0.0)), boundaries, 6)
    d_barwall = DirichletBC(VVQ.sub(1), ((0.0, 0.0)), boundaries, 7) #No slip on geometry in fluid

    #Pressure Conditions
    p_out = DirichletBC(VVQ.sub(2), 0, boundaries, 4)

    #Assemble boundary conditions
    bcs = [u_inlet, u_wall, u_circ, u_barwall,\
           d_wall, d_inlet, d_outlet, d_circle,d_barwall,\
           p_out]#,p_bar]
    return bcs,dx_f,dx_s,ds,dS,n,inlet,coord,boundaries
