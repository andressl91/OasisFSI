from dolfin import *
import numpy as np
import matplotlib.pyplot as plt
import sys

mesh_file = Mesh("Mesh/fluid_new.xml")
#mesh_file = refine(mesh_file)
#Parameters for each numerical case
common = {"mesh": mesh_file,
          "v_deg": 2,    #Velocity degree
          "p_deg": 1,    #Pressure degree
          "d_deg": 2,    #Deformation degree
          "T": 40,          # End time
          "dt": 0.001,       # Time step
          "rho_f": 1.0E3,    #
          "mu_f": 1.0,
          "rho_s" : Constant(10E3),
          "mu_s" : Constant(0.5E6),
          "nu_s" : Constant(0.4),
          "Um" : 1.0,
          "D" : 0.1,
          "H" : 0.41,
          "L" : 2.5,
    	  "step" : 10,
          "checkpoint": False
          }
#"checkpoint": "./FSI_fresh_checkpoints/FSI-2/P-2/dt-0.05/dvpFile.h5"

vars().update(common)
lamda_s = nu_s*2*mu_s/(1 - 2.*nu_s)
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
n = FacetNormal(mesh_file)

Bar_area = AutoSubDomain(lambda x: (0.19 <= x[1] <= 0.21) and 0.24<= x[0] <= 0.6) # only the "flag" or "bar"
domains = CellFunction("size_t", mesh_file)
domains.set_all(1)
Bar_area.mark(domains, 2) #Overwrites structure domain
dx = Measure("dx", subdomain_data = domains)
#plot(domains,interactive = True)
dx_f = dx(1, subdomain_data = domains)
dx_s = dx(2, subdomain_data = domains)
dis_x = []
dis_y = []
Drag_list = []
Lift_list = []

hdf = HDF5File(mesh.mpi_comm(), "./FSI_fresh_checkpoints/FSI3/dvp.h5", "w")
#Fluid properties

class Inlet(Expression):
    def __init__(self, Um):
        self.t = 0
        self.Um = Um
    def eval(self, value, x):
    	value[0] = 0.5*(1-np.cos(self.t*np.pi/2))*1.5*self.Um*x[1]*(H-x[1])/((H/2.0)**2)
    	value[1] = 0
    def value_shape(self):
    	return (2,)

inlet = Inlet(Um)
u_file = XDMFFile(mpi_comm_world(), "FSI_fresh_results/FSI-2/P-"+str(v_deg) +"/dt-"+str(dt)+"/velocity.xdmf")
d_file = XDMFFile(mpi_comm_world(), "FSI_fresh_results/FSI-2/P-"+str(v_deg) +"/dt-"+str(dt)+"/d.xdmf")
p_file = XDMFFile(mpi_comm_world(), "FSI_fresh_results/FSI-2/P-"+str(v_deg) +"/dt-"+str(dt)+"/pressure.xdmf")

for tmp_t in [u_file, d_file, p_file]:
    tmp_t.parameters["flush_output"] = True
    tmp_t.parameters["multi_file"] = 0
    tmp_t.parameters["rewrite_function_mesh"] = False
if checkpoint == "FSI_fresh_checkpoints/FSI-2/P-"+str(v_deg)+"/dt-"+str(dt)+"/dvpFile.h5":
    print "test"
    sys.exit(0)
else:
    dvp_file=HDF5File(mpi_comm_world(), "FSI_fresh_checkpoints/FSI-2/P-"+str(v_deg)+"/dt-"+str(dt)+"/dvpFile.h5", "w")


def initiate(v_deg, dt, **monolithic):

    return {}

def create_bcs(DVP, dvp_, n, k, Um, H, boundaries, inlet, **semimp_namespace):
    #Fluid velocity conditions
    u_inlet  = DirichletBC(DVP.sub(1), inlet, boundaries, 3)
    u_wall   = DirichletBC(DVP.sub(1), ((0.0, 0.0)), boundaries, 2)
    u_circ   = DirichletBC(DVP.sub(1), ((0.0, 0.0)), boundaries, 6) #No slip on geometry in fluid
    u_barwall= DirichletBC(DVP.sub(1), ((0.0, 0.0)), boundaries, 7) #No slip on geometry in fluid

    #displacement conditions:
    d_wall    = DirichletBC(DVP.sub(0), ((0.0, 0.0)), boundaries, 2)
    d_inlet   = DirichletBC(DVP.sub(0), ((0.0, 0.0)), boundaries, 3)
    d_outlet  = DirichletBC(DVP.sub(0), ((0.0, 0.0)), boundaries, 4)
    d_circle  = DirichletBC(DVP.sub(0), ((0.0, 0.0)), boundaries, 6)
    d_barwall = DirichletBC(DVP.sub(0), ((0.0, 0.0)), boundaries, 7) #No slip on geometry in fluid

    #Pressure Conditions
    p_out = DirichletBC(DVP.sub(2), 0, boundaries, 4)

    #Assemble boundary conditions
    bcs = [u_inlet, u_wall, u_circ, u_barwall,\
           d_wall, d_inlet, d_outlet, d_circle,d_barwall,\
           p_out]

    return dict(bcs = bcs)


def pre_solve(t, inlet, **semimp_namespace):
    if t < 2:
        inlet.t = t
    else:
        inlet.t = 2

    return dict(inlet = inlet)


#dvp_file = XDMFFile(mpi_comm_world(), "FSI_fresh_checkpoints/FSI-2/P-"+str(v_deg)+"/dt-"+str(dt)+"/dvpFile.xdmf")


def after_solve(t, dvp_, coord,dis_x,dis_y,Drag_list,Lift_list,counter,dvp_file,u_file,p_file,d_file, **semimp_namespace):

    d, v, p = dvp_["n"].split(True)
    if counter%step ==0:
        #u_file << v
        #d_file << d
        #p_file << p
        #p_file.write(p)
        d_file.write(d)
        u_file.write(v)
        #dvp_file << dvp_["n"]
        dvp_file.write(dvp_["n"], "dvp%g"%t)

    def F_(U):
        return (Identity(len(U)) + grad(U))

    def J_(U):
        return det(F_(U))

    def sigma_f_new(v, p, d, mu_f):
        return -p*Identity(len(v)) + mu_f*(grad(v)*inv(F_(d)) + inv(F_(d)).T*grad(v).T)

    Dr = -assemble((sigma_f_new(v,p,d,mu_f)*n)[0]*ds(6))
    Li = -assemble((sigma_f_new(v,p,d,mu_f)*n)[1]*ds(6))
    Dr += -assemble((sigma_f_new(v("-"),p("-"),d("-"),mu_f)*n("-"))[0]*dS(5))
    Li += -assemble((sigma_f_new(v("-"),p("-"),d("-"),mu_f)*n("-"))[1]*dS(5))
    Drag_list.append(Dr)
    Lift_list.append(Li)


    dsx = d(coord)[0]
    dsy = d(coord)[1]
    dis_x.append(dsx)
    dis_y.append(dsy)
    if MPI.rank(mpi_comm_world()) == 0:
        print "LIFT = %g,  DRAG = %g" % (Li, Dr)
        print "dis_x/dis_y : %g %g "%(dsx,dsy)

    return {}


def post_process(T,dt,dis_x,dis_y, Drag_list,Lift_list,dvp_file,**semimp_namespace):
    dvp_file.close()
    """
    time_list = np.linspace(0,T,T/dt+1)
    plt.plot(time_list,dis_x); plt.ylabel("Displacement x");plt.xlabel("Time");plt.grid();
    #plt.savefig("FSI_results/FSI-1/P-"+str(v_deg) +"/dt-"+str(dt)+"/dis_x.png")
    plt.show()
    plt.plot(time_list,dis_y);plt.ylabel("Displacement y");plt.xlabel("Time");plt.grid();
    #plt.savefig("FSI_results/FSI-1/P-"+str(v_deg) +"/dt-"+str(dt)+"/dis_y.png")
    plt.show()
    plt.plot(time_list,Drag);plt.ylabel("Drag");plt.xlabel("Time");plt.grid();
    #plt.savefig("FSI_results/FSI-1/P-"+str(v_deg) +"/dt-"+str(dt)+"/drag.png")
    plt.show()
    plt.plot(time_list,Lift);plt.ylabel("Lift");plt.xlabel("Time");plt.grid();
    #plt.savefig("FSI_results/FSI-1/P-"+str(v_deg) +"/dt-"+str(dt)+"/lift.png")
    plt.show()
    """
    return {}

def initiate(dvp_, checkpoint,t, **monolithic):
    print checkpoint
    if checkpoint != False:
        hdf = HDF5File(mpi_comm_world(), checkpoint, "r")
        print hdf.has_dataset("/dvp4")
        hdf.read(dvp_["n-1"], "/dvp4")
        t = 4.0
        #print "Type: ",type(dvp_["n-1"])
        #d, u, p = dvp_["n-1"].split(True)
        #f << u
       	#plot(u, interactive=True)
    return dict(t=t)
