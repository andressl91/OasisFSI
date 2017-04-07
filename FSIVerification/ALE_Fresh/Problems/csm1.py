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
          "T": 10.0,          # End time
          "dt": 0.1,       # Time step
          "rho_f": 1.0E3,    #
          "mu_f": 1.0,
          "rho_s" : Constant(1.0E3),
          "mu_s" : Constant(0.5E6),
          "nu_s" : Constant(0.4),
          "Um" : 0.0,
          "D" : 0.1,
          "H" : 0.41,
          "L" : 2.5,
    	  "step" : 1,
          "checkpoint": False
          }
 #"checkpoint": "./FSI_fresh_checkpoints/CSM-1/P-2/dt-0.05/dvpFile.h5"
vars().update(common)
lamda_s = nu_s*2*mu_s/(1 - 2.*nu_s)
#plot(mesh, interactive=True)
sebb = 69
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
Time_list = []
Det_list = []
Det_list_mean = []
Det_list_max = []


#dvp_file = XDMFFile(mpi_comm_world(), "FSI_fresh_checkpoints/CSM-1/P-"+str(v_deg)+"/dt-"+str(dt)+"/dvpFile.xdmf")


if checkpoint == "FSI_fresh_checkpoints/CSM-1/P-"+str(v_deg)+"/dt-"+str(dt)+"/dvpFile.h5":
    sys.exit(0)
else:
    dvp_file=HDF5File(mpi_comm_world(), "FSI_fresh_checkpoints/CSM-1/P-"+str(v_deg)+"/dt-"+str(dt)+"/dvpFile.h5", "w")


def initiate(F_solid_linear, mesh_file, rho_s, psi, extype, dx_s, v_deg, dt, P, dvp_, **semimp_namespace):

    #gravity = Constant((0, -2*rho_s))
    #F_solid_linear -= inner(gravity, psi)*dx_s
    def F_(U):
        return  grad(U)

    def J_(U):
        return det(F_(U))

    u_file = XDMFFile(mpi_comm_world(), "CSM_results/CSM-1/"+str(extype) +"/dt-"+str(dt)+"/velocity.xdmf")
    d_file = XDMFFile(mpi_comm_world(), "CSM_results/CSM-1/"+str(extype) +"/dt-"+str(dt)+"/d.xdmf")
    det_file = XDMFFile(mpi_comm_world(), "CSM_results/CSM-1/"+str(extype) +"/dt-"+str(dt)+"/det.xdmf")
    for tmp_t in [u_file, d_file, det_file]:
        tmp_t.parameters["flush_output"] = True
        tmp_t.parameters["multi_file"] = 0
        tmp_t.parameters["rewrite_function_mesh"] = False

    d, v, p = dvp_["n"].split(True)
    d_file.write(d)
    u_file.write(v)


    #dg = FunctionSpace(mesh_file, "DG", 0)
    det_func = Function(P)
    Det = project(J_(d), P)
    det_func.vector().zero()
    det_func.vector().axpy(1, Det.vector())
    det_func.rename("det", "det")
    det_file.write(det_func)
    Det_list.append((det_func.vector().array()).min())
    Det_list_max.append((det_func.vector().array()).max())
    Det_list_mean.append(np.mean(det_func.vector().array()))

    return dict(u_file=u_file, d_file=d_file, det_file=det_file,F_solid_linear=F_solid_linear, det_func=det_func)

def create_bcs(DVP, dvp_, n, k, Um, H, boundaries,  **semimp_namespace):
    print "Create bcs"
    #Fluid velocity conditions
    u_inlet  = DirichletBC(DVP.sub(1), ((0.0, 0.0)), boundaries, 3)
    u_wall   = DirichletBC(DVP.sub(1), ((0.0, 0.0)), boundaries, 2)
    #u_outlet = DirichletBC(DVP.sub(1), ((0.0, 0.0)), boundaries, 4)
    u_circ   = DirichletBC(DVP.sub(1), ((0.0, 0.0)), boundaries, 6) #No slip on geometry in fluid
    u_barwall= DirichletBC(DVP.sub(1), ((0.0, 0.0)), boundaries, 7) #No slip on geometry in fluid

    #displacement conditions:
    d_wall    = DirichletBC(DVP.sub(0), ((0.0, 0.0)), boundaries, 2)
    d_inlet   = DirichletBC(DVP.sub(0), ((0.0, 0.0)), boundaries, 3)
    d_outlet  = DirichletBC(DVP.sub(0), ((0.0, 0.0)), boundaries, 4)
    d_circle  = DirichletBC(DVP.sub(0), ((0.0, 0.0)), boundaries, 6)
    d_barwall = DirichletBC(DVP.sub(0), ((0.0, 0.0)), boundaries, 7) #No slip on geometry in fluid


    #Assemble boundary conditions
    bcs = [u_wall, u_inlet, u_circ, u_barwall,\
           d_wall, d_inlet, d_outlet, d_circle,d_barwall]

    return dict(bcs = bcs)


def pre_solve(**semimp_namespace):
    return {}


def after_solve(t, det_func, P, DVP, dvp_, n,coord,dis_x,dis_y,Drag_list,Lift_list, Det_list,\
                det_file, Det_list_max, Det_list_mean, counter,dvp_file,u_file,d_file, **semimp_namespace):

    d, v, p = dvp_["n"].split(True)
    if counter%step ==0:
        #u_file << v
        #d_file << d
        #p_file << p
        #p_file.write(p)
        d_file.write(d)
        u_file.write(v)
        #dvp_file << dvp_["n"]
        #dvp_file.write(dvp_["n"], "dvp%g"%t)

    def F_(U):
        return grad(U)

    def J_(U):
        return det(F_(U))

    def sigma_f_new(v, p, d, mu_f):
        return -p*Identity(len(v)) + mu_f*(grad(v)*inv(F_(d)) + inv(F_(d)).T*grad(v).T)

    #Det = project(J_(d), DVP.sub(0).collapse())
    Det = project(J_(d), P)
    det_func.vector().zero()
    det_func.vector().axpy(1, Det.vector())
    det_func.rename("det", "det")
    det_file.write(det_func)
    Det_list.append((det_func.vector().array()).min())
    Det_list_max.append((det_func.vector().array()).max())
    Det_list_mean.append(np.mean(det_func.vector().array()))

    Time_list.append(t)
    dsx = d(coord)[0]
    dsy = d(coord)[1]
    dis_x.append(dsx)
    dis_y.append(dsy)
    if MPI.rank(mpi_comm_world()) == 0:
        print "dis_x/dis_y : %g %g "%(dsx,dsy)

    return {}


def post_process(T, Det_list_mean, Det_list_max, extype,dt,Det_list,dis_x,dis_y, Drag_list,Lift_list, Time_list, dvp_file,**semimp_namespace):
    plt.figure(1)
    plt.plot(Time_list,dis_x); plt.ylabel("Displacement x");plt.xlabel("Time");plt.grid();
    plt.savefig("CSM_results/CSM-1/"+str(extype) +"/dt-"+str(dt)+"/dis_x.png")
    plt.figure(2)
    plt.plot(Time_list,dis_y);plt.ylabel("Displacement y");plt.xlabel("Time");plt.grid();
    plt.savefig("CSM_results/CSM-1/"+str(extype) +"/dt-"+str(dt)+"/dis_y.png")
    plt.figure(5)
    Time_list = [0] + Time_list
    plt.plot(Time_list,Det_list);plt.ylabel("Min_Det(F)");plt.xlabel("Time");plt.grid();
    plt.savefig("CSM_results/CSM-1/"+str(extype) +"/dt-"+str(dt)+"/Min_J.png")
    plt.figure(3)
    plt.plot(Time_list,Det_list_mean);plt.ylabel("Mean_Det(F)");plt.xlabel("Time");plt.grid();
    plt.savefig("CSM_results/CSM-1/"+str(extype) +"/dt-"+str(dt)+"/Mean_J.png")
    plt.figure(4)
    plt.plot(Time_list,Det_list_max);plt.ylabel("Max_Det(F)");plt.xlabel("Time");plt.grid();
    plt.savefig("CSM_results/CSM-1/"+str(extype) +"/dt-"+str(dt)+"/Max_J.png")

    return {}
