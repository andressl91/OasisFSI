from dolfin import *
import numpy as np
import matplotlib.pyplot as plt
import sys



mesh_file = RectangleMesh(Point(0.0,0.0), Point(10,2.2), 50, 25, "right/left")
#plot(mesh_file,interactive=True , title="Rectangle (right/left)")

#Parameters for each numerical case
common = {"mesh": mesh_file,
          "v_deg": 2,    #Velocity degree
          "p_deg": 1,    #Pressure degree
          "d_deg": 2,    #Deformation degree
          "T": 0.002,          # End time
          "dt": 0.001,       # Time step
          "rho_f": 1.0E3,    #
          "mu_f": 4.0E-3,
          "rho_s" : Constant(1.0E3),
          "mu_s" : Constant(385.0E3),
          "nu_s" : Constant(0.3),
          "Um" : 2.0,
    	  "step" : 1,
          "checkpoint": False
          }
 #"checkpoint": "./FSI_fresh_checkpoints/tube/P-2/dt-0.05/dvpFile.h5"
vars().update(common)
lamda_s = nu_s*2*mu_s/(1 - 2.*nu_s)
#plot(mesh, interactive=True)


# BOUNDARIES

#NOS = AutoSubDomain(lambda x: "on_boundary" and( near(x[1],0) or near(x[1], 0.41)))
Inlet = AutoSubDomain(lambda x: "on_boundary" and (near(x[0],0) and x[1]<2.0))
Outlet = AutoSubDomain(lambda x: "on_boundary" and (near(x[0],10) and x[1]<2.0))
Floor =  AutoSubDomain(lambda x: "on_boundary" and near(x[1], 0.0))
#Inlet_Outlet_S = AutoSubDomain(lambda x: "on_boundary" and ((near(x[0],0) or near(x[0],10)) and x[1]>2.0 ))
Inlet_Outlet_S = AutoSubDomain(lambda x: "on_boundary" and (near(x[0],0)) or near(x[0],10))

#Top = AutoSubDomain(lambda x: "on_boundary" and near(x[1], 2.0))

Allboundaries = DomainBoundary()

boundaries = FacetFunction("size_t",mesh_file)
boundaries.set_all(0)
Inlet_Outlet_S.mark(boundaries,5)
Inlet.mark(boundaries, 3)
Floor.mark(boundaries, 2)
Outlet.mark(boundaries, 4)


#plot(boundaries,interactive=True)

ds = Measure("ds", subdomain_data = boundaries)
dS = Measure("dS", subdomain_data = boundaries)
n = FacetNormal(mesh_file)

Bar_area = AutoSubDomain(lambda x: (2.2 >= x[1] >= 2.0)) # only the "flag" or "bar"
domains = CellFunction("size_t", mesh_file)
domains.set_all(1)
Bar_area.mark(domains, 2) #Overwrites structure domain
dx = Measure("dx", subdomain_data = domains)
#plot(domains,interactive = True)

dx_f = dx(1, subdomain_data = domains)
dx_s = dx(2, subdomain_data = domains)

Time_list = []
Det_list = []




#dvp_file = XDMFFile(mpi_comm_world(), "FSI_fresh_checkpoints/tube/P-"+str(v_deg)+"/dt-"+str(dt)+"/dvpFile.xdmf")



def initiate(v_deg, dt, theta, dvp_, args, **semimp_namespace):
    if args.extravari == "alfa":
        path =  "FSI_fresh_results/tube/"+str(args.extravari) +"_"+ str(args.extype) +"/dt-"+str(dt)+"_theta-"+str(theta)
    if args.extravari == "biharmonic":
        path = "FSI_fresh_results/tube/"+str(args.extravari) +"/dt-"+str(dt)+"_theta-"+str(theta)

    u_file = XDMFFile(mpi_comm_world(), path + "/velocity.xdmf")
    d_file = XDMFFile(mpi_comm_world(), path + "/d.xdmf")
    p_file = XDMFFile(mpi_comm_world(), path + "/pressure.xdmf")
    for tmp_t in [u_file, d_file, p_file]:
        tmp_t.parameters["flush_output"] = True
        tmp_t.parameters["multi_file"] = 0
        tmp_t.parameters["rewrite_function_mesh"] = False
    #d = dvp_["n"].sub(0, deepcopy=True)
    #v = dvp_["n"].sub(1, deepcopy=True)
    #p = dvp_["n"].sub(2, deepcopy=True)
    #p_file.write(p)
    #d_file.write(d)
    #u_file.write(v)
    #p_file << p
    #d_file << d
    #u_file << v

    return dict(u_file=u_file, d_file=d_file, p_file=p_file, path=path)

def create_bcs(DVP, dvp_, n, k, boundaries, **semimp_namespace):
    print "Create bcs"
    #Fluid velocity conditions
    u_inlet  = DirichletBC(DVP.sub(1), ((0.0, 0.0)), boundaries, 3)
    u_floor  = DirichletBC(DVP.sub(1), ((0.0, 0.0)), boundaries, 2)
    u_outlet = DirichletBC(DVP.sub(1), ((0.0, 0.0)), boundaries, 4) #No slip on geometry in fluid
    u_inlet_outlet  = DirichletBC(DVP.sub(1), ((0.0, 0.0)), boundaries, 5)

    #displacement conditions:
    d_inlet_outlet  = DirichletBC(DVP.sub(0), ((0.0, 0.0)), boundaries, 5)
    d_inlet   = DirichletBC(DVP.sub(0), ((0.0, 0.0)), boundaries, 3)
    d_outlet  = DirichletBC(DVP.sub(0), ((0.0, 0.0)), boundaries, 4)
    d_floor   = DirichletBC(DVP.sub(0), ((0.0, 0.0)), boundaries, 2) #No slip on geometry in fluid

    #Pressure Conditions
    p_inlet = DirichletBC(DVP.sub(2), 5000, boundaries, 3)
    p_out = DirichletBC(DVP.sub(2), 0, boundaries, 4)

    #Assemble boundary conditions
    bcs = [u_floor, u_inlet_outlet,\
           d_inlet, d_floor,d_outlet,d_inlet_outlet,\
           p_inlet, p_out]

    return dict(bcs = bcs)


def pre_solve(**semimp_namespace):

    return {}


def after_solve(t, P, DVP, dvp_, n, Det_list,\
                counter,u_file,p_file,d_file, **semimp_namespace):

    d = dvp_["n"].sub(0, deepcopy=True)
    v = dvp_["n"].sub(1, deepcopy=True)
    p = dvp_["n"].sub(2, deepcopy=True)
    #d, v, p = dvp_["n"].split(True)
    if counter%step ==0:
        #u_file << v
        #d_file << d
        #p_file << p
        p_file.write(p)
        d_file.write(d)
        u_file.write(v)
        #dvp_file << dvp_["n"]
        #dvp_file.write(dvp_["n"], "dvp%g"%t)
    #plot(p,interactive=True)
    #plot(v,interactive=True)

    def F_(U):
        return (Identity(len(U)) + grad(U))

    def J_(U):
        return det(F_(U))

    #Det = project(J_(d), DVP.sub(0).collapse())
    Det = project(J_(d), P)
    Det_list.append((Det.vector().array()).min())


    return {}


def post_process(path,T,dt,Det_list,dis_x,dis_y, Time_list,\
                args, simtime,v_deg, p_deg, d_deg, dvp_file,**semimp_namespace):
    #dvp_file.close()
    #time_list = np.linspace(0,T,T/dt+1)
    theta = args.theta
    f_scheme = args.fluidvari
    s_scheme = args.solidvari
    e_scheme = args.extravari
    f = open(path+"/report.txt", 'w')
    f.write("""FSI3 EXPERIMENT
    T = %(T)g\ndt = %(dt)g\nv_deg = %(d_deg)g\nv_deg = %(v_deg)g\np_deg = %(p_deg)g\n
theta = %(theta)s\nf_vari = %(f_scheme)s\ns_vari = %(s_scheme)s\ne_vari = %(e_scheme)s\n time = %(simtime)g""" % vars())
    #f.write("""Runtime = %f """ % fintime)
    f.close()

    np.savetxt(path + '/Lift.txt', Lift_list, delimiter=',')
    np.savetxt(path + '/Drag.txt', Drag_list, delimiter=',')
    np.savetxt(path + '/Time.txt', Time_list, delimiter=',')
    np.savetxt(path + '/dis_x.txt', dis_x, delimiter=',')
    np.savetxt(path + '/dis_y.txt', dis_y, delimiter=',')

    plt.figure(1)
    plt.plot(Time_list,dis_x); plt.ylabel("Displacement x");plt.xlabel("Time");plt.grid();
    plt.savefig(path + "/dis_x.png")
    plt.figure(2)
    plt.plot(Time_list,dis_y);plt.ylabel("Displacement y");plt.xlabel("Time");plt.grid();
    plt.savefig(path + "/dis_y.png")
    plt.figure(3)
    plt.plot(Time_list,Drag_list);plt.ylabel("Drag");plt.xlabel("Time");plt.grid();
    plt.savefig(path + "/drag.png")
    plt.figure(4)
    plt.plot(Time_list,Lift_list);plt.ylabel("Lift");plt.xlabel("Time");plt.grid();
    plt.savefig(path + "/lift.png")
    plt.figure(5)
    plt.plot(Time_list,Det_list);plt.ylabel("Min_Det(F)");plt.xlabel("Time");plt.grid();
    plt.savefig(path + "/Min_J.png")

    return {}
