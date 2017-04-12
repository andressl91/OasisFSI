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
print "GALA"
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

#dvp_file = XDMFFile(mpi_comm_world(), "FSI_fresh_checkpoints/CSM-1/P-"+str(v_deg)+"/dt-"+str(dt)+"/dvpFile.xdmf")


if checkpoint == "FSI_fresh_checkpoints/CSM-1/P-"+str(v_deg)+"/dt-"+str(dt)+"/dvpFile.h5":
    sys.exit(0)
else:
    dvp_file=HDF5File(mpi_comm_world(), "FSI_fresh_checkpoints/CSM-1/P-"+str(v_deg)+"/dt-"+str(dt)+"/dvpFile.h5", "w")


def initiate(t, F_solid_linear, args, theta, mesh_file, rho_s, psi, extype, dx_s, v_deg, dt, P, dvp_, Time_list, Det_list,**semimp_namespace):

    #gravity = Constant((0, -2*rho_s))
    #F_solid_linear -= inner(gravity, psi)*dx_s
    def F_(U):
        return Identity(len(U)) +  grad(U)

    def J_(U):
        return det(F_(U))

    if args.extravari == "alfa":
        path = "CSM_results/CSM-1/"+str(args.extravari) +"_"+ str(args.extype) +"/dt-"+str(dt)+"_theta-"+str(theta)
    if args.extravari == "biharmonic":
        path = "CSM_results/CSM-1/"+str(args.extravari) +"/dt-"+str(dt)+"_theta-"+str(theta)

    u_file = XDMFFile(mpi_comm_world(), path + "/velocity.xdmf")
    d_file = XDMFFile(mpi_comm_world(), path + "/d.xdmf")
    p_file = XDMFFile(mpi_comm_world(), path + "/pressure.xdmf")
    for tmp_t in [u_file, d_file, p_file]:
        tmp_t.parameters["flush_output"] = True
        tmp_t.parameters["multi_file"] = 0
        tmp_t.parameters["rewrite_function_mesh"] = False

    d = dvp_["n"].sub(0, deepcopy=True)
    v = dvp_["n"].sub(1, deepcopy=True)
    d_file.write(d)
    u_file.write(v)

    #dg = FunctionSpace(mesh_file, "DG", 0)
    det_func = Function(P)
    Det = project(J_(d), P)
    det_func.vector().zero()
    det_func.vector().axpy(1, Det.vector())

    Time_list.append(t)
    dsx = d(coord)[0]
    dsy = d(coord)[1]
    dis_x.append(dsx)
    dis_y.append(dsy)

    Det_list.append((det_func.vector().array()).min())


    return dict(u_file=u_file, d_file=d_file, det_func=det_func, path=path)

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

    p_outlet  = DirichletBC(DVP.sub(2), (0.0), boundaries, 4)

    #Assemble boundary conditions
    bcs = [u_wall, u_inlet, u_circ, u_barwall,\
           d_wall, d_inlet, d_outlet, d_circle,d_barwall,\
           p_outlet]

    return dict(bcs = bcs)


def pre_solve(**semimp_namespace):
    return {}


def after_solve(t, det_func, P, DVP, dvp_, n,coord,dis_x,dis_y, Det_list,\
                counter,dvp_file,u_file,d_file, **semimp_namespace):

    d = dvp_["n"].sub(0, deepcopy=True)
    v = dvp_["n"].sub(1, deepcopy=True)
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
        return Identity(len(U)) +  grad(U)

    def J_(U):
        return det(F_(U))

    def sigma_f_new(v, p, d, mu_f):
        return -p*Identity(len(v)) + mu_f*(grad(v)*inv(F_(d)) + inv(F_(d)).T*grad(v).T)

    #Det = project(J_(d), DVP.sub(0).collapse())
    Det = project(J_(d), P)
    det_func.vector().zero()
    det_func.vector().axpy(1, Det.vector())
    Det_list.append((det_func.vector().array()).min())

    Time_list.append(t)
    dsx = d(coord)[0]
    dsy = d(coord)[1]
    dis_x.append(dsx)
    dis_y.append(dsy)
    if MPI.rank(mpi_comm_world()) == 0:
        print "dis_x/dis_y : %g %g "%(dsx,dsy)

    return {}


def post_process(path,T,dt,Det_list,dis_x,dis_y, Time_list,\
                    args, v_deg, p_deg, d_deg, **semimp_namespace):

    theta = args.theta
    f_scheme = args.fluidvari
    s_scheme = args.solidvari
    e_scheme = args.extravari
    f = open(path+"/report.txt", 'w')
    f.write("""FSI3 EXPERIMENT
    T = %(T)g\ndt = %(dt)g\nv_deg = %(d_deg)g\nv_deg = %(v_deg)g\np_deg = %(p_deg)g\n
theta = %(theta)s\nf_vari = %(f_scheme)s\ns_vari = %(s_scheme)s\ne_vari = %(e_scheme)s\n""" % vars())
    #f.write("""Runtime = %f """ % fintime)
    f.close()

    np.savetxt(path + '/time.txt', Time_list, delimiter=',')
    np.savetxt(path + '/dis_x.txt', dis_x, delimiter=',')
    np.savetxt(path + '/dis_y.txt', dis_y, delimiter=',')
    np.savetxt(path + '/min_J.txt', Det_list, delimiter=',')

    plt.figure(1)
    plt.plot(Time_list,dis_x); plt.ylabel("Displacement x");plt.xlabel("Time");plt.grid();
    plt.savefig(path + "/dis_x.png")
    plt.figure(2)
    plt.plot(Time_list,dis_y);plt.ylabel("Displacement y");plt.xlabel("Time");plt.grid();
    plt.savefig(path + "/dis_y.png")
    plt.figure(3)
    plt.plot(Time_list,Det_list);plt.ylabel("Min_Det(F)");plt.xlabel("Time");plt.grid();
    plt.savefig(path + "/Min_J.png")

    return {}
