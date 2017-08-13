from dolfin import *
import os
import numpy as np
import matplotlib.pyplot as plt

# TODO: There should not be any global variable in this file

refi = 0
mesh_name = "base0"
mesh_file = Mesh("Mesh/" + mesh_name +".xml")
#mesh_file = refine(mesh_file)
#Parameters for each numerical case
common = {"mesh": mesh_file,
          "v_deg": 2,    #Velocity degree
          "p_deg": 1,    #Pressure degree
          "d_deg": 2,    #Deformation degree
          "T": 8,          # End time
          #"dt": 0.5,       # Time step
          "rho_f": 1.0E3,    #
          "mu_f": 1.,
          "rho_s" : Constant(1.0E3),
          "mu_s" : Constant(0.5E6),
          "nu_s" : Constant(0.4),
          "Um" : 0.2,
          "D" : 0.1,
          "H" : 0.41,
          "L" : 2.5,
          "checkpoint": 1,
          "step": 1
     }

#"checkpoint": "./FSI_fresh_checkpoints/FSI-1/P-2/dt-0.05/dvpFile.h5"
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

boundaries = FacetFunction("size_t", mesh_file)
boundaries.set_all(0)
Allboundaries.mark(boundaries, 1)
Wall.mark(boundaries, 2)
Inlet.mark(boundaries, 3)
Outlet.mark(boundaries, 4)
Bar.mark(boundaries, 5)
Circle.mark(boundaries, 6)
Barwall.mark(boundaries, 7)
#plot(boundaries,interactive=True)

ds2 = Measure("ds", subdomain_data = boundaries)
dS2 = Measure("dS", subdomain_data = boundaries)

n = FacetNormal(mesh_file)

Bar_area = AutoSubDomain(lambda x: (0.19 <= x[1] <= 0.21) and 0.24<= x[0] <= 0.6) # only the "flag" or "bar"
domains = CellFunction("size_t", mesh_file)
domains.set_all(1)
Bar_area.mark(domains, 2) #Overwrites structure domain

dx = Measure("dx", subdomain_data = domains)
dx_f = dx(1, subdomain_data = domains)
dx_s = dx(2, subdomain_data = domains)

dis_x = []
dis_y = []
Drag_list = []
Lift_list = []
Time_list = []
Det_list = []

body_force = Constant((0.0, 0.0))

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

def initiate(v_deg, V, d_deg, p_deg, dt, theta, vpd_, args, mesh_name, refi, **semimp_namespace):
    # Set path based on solver parameters
    exva = args.extravari
    extype = args.extype
    bitype = args.bitype
    path = "FSI_fresh_results/FSI-1/%(exva)s_%(extype)s/dt-%(dt)g_theta-%(theta)g/%(mesh_name)s_refine_%(refi)d_v_deg_%(v_deg)s_d_deg_%(d_deg)s_p_deg_%(p_deg)s" % vars()

    # Create folders if they do not exists
    if MPI.rank(mpi_comm_world()) == 0:
        if not os.path.exists(path):
            os.makedirs(path)
    MPI.barrier(mpi_comm_world())

    # Files to store results
    u_file = XDMFFile(mpi_comm_world(), path + "/velocity.xdmf")
    v_tilde_file = XDMFFile(mpi_comm_world(), path + "/velocity_tilde.xdmf")
    d_file = XDMFFile(mpi_comm_world(), path + "/d.xdmf")
    dtilde_file = XDMFFile(mpi_comm_world(), path + "/d_tilde.xdmf")
    p_file = XDMFFile(mpi_comm_world(), path + "/pressure.xdmf")
    w_file = XDMFFile(mpi_comm_world(), path + "/mesh_velocity.xdmf")
    for tmp_t in [u_file, d_file, p_file, dtilde_file, v_tilde_file, w_file]:
        tmp_t.parameters["flush_output"] = True
        tmp_t.parameters["multi_file"] = 0
        tmp_t.parameters["rewrite_function_mesh"] = False

    # Function to compute w_tilde (for testing)
    w_test = Function(V)

    return dict(u_file=u_file, d_file=d_file, p_file=p_file, dtilde_file=dtilde_file,
                v_tilde_file=v_tilde_file, w_file=w_file, path=path,
                w_test=w_test)

def create_bcs(vpd_, VPD, D, V, args, boundaries, inlet, dt, **semimp_namespace):
    # d_bc_n, d_bc_n1
    print "Create bcs"

    noslip = ((0.0, 0.0))
    wm_inlet  = DirichletBC(D, noslip, boundaries, 3)
    wm_outlet = DirichletBC(D, noslip, boundaries, 4)
    wm_wall   = DirichletBC(D, noslip, boundaries, 2)
    wm_circ   = DirichletBC(D, noslip, boundaries, 6)
    wm_bar    = DirichletBC(D, vpd_["tilde"].sub(2), boundaries, 5)
    bcs_w = [wm_wall, wm_inlet, wm_outlet, wm_circ, wm_bar]

    # Fluid tentative bcs
    #d_tilde = dw_["tilde"].sub(0)
    #d_n1 = dw_["n-1"].sub(0)
    #w_bar_t = (d_tilde - d_n1) / k

    class W_bar(Expression):
        def eval(self, value, x):
            value[:] = (vpd_["tilde"].sub(2)(x) - vpd_["n-1"].sub(2)(x)) / dt
            #if x[0] == 0.6 and x[1] == 0.2:
            #    print value, x

        def value_shape(self):
            return (2, )

    class W_bar2(Expression):
        def eval(self, value, x):
            value[:] = (vpd_["n"].sub(2)(x) - vpd_["n-1"].sub(2)(x)) / dt
            #if x[0] == 0.6 and x[1] == 0.2:
            #    print value, x

        def value_shape(self):
            return (2, )

    w_bar = W_bar()
    w_bar2 = W_bar2()

    # Tentative velocity
    u_inlet_t = DirichletBC(V, inlet, boundaries, 3)
    u_wall_t = DirichletBC(V, noslip, boundaries, 2)
    u_circ_t = DirichletBC(V, noslip, boundaries, 6)
    u_bar_t = DirichletBC(V, w_bar, boundaries, 5)

    bcs_tent = [u_wall_t, u_inlet_t, u_circ_t, u_bar_t]

    # Coupled
    u_inlet = DirichletBC(VPD.sub(0), inlet, boundaries, 3)
    u_wall = DirichletBC(VPD.sub(0), noslip, boundaries, 2)
    u_circ = DirichletBC(VPD.sub(0), noslip, boundaries, 6)
    #u_bar = DirichletBC(VPD.sub(0), w_bar2, boundaries, 5)

    p_outlet  = DirichletBC(VPD.sub(1), (0), boundaries, 4)

    d_barwall = DirichletBC(VPD.sub(2), noslip, boundaries, 7)

    bcs_coupled = [u_wall, u_inlet, u_circ, p_outlet, d_barwall] #, u_bar]
    return dict(bcs_tent=bcs_tent, bcs_w=bcs_w, bcs_coupled=bcs_coupled)


def pre_solve(t, inlet, **semimp_namespace):
    if t < 2:
        inlet.t = t
    else:
        inlet.t = 2

    return dict(inlet = inlet)


def after_solve(t, P, vpd_, n, coord, dis_x, dis_y, Drag_list, Lift_list, Det_list,\
                dtilde_file, counter, u_file, p_file, d_file, v_tilde_file,
                w_file, w_test, dt, **semimp_namespace):

    d_tilde = vpd_["tilde"].sub(0, deepcopy=True)
    v = vpd_["n"].sub(0, deepcopy=True)
    v_tilde = vpd_["tilde"].sub(0, deepcopy=True)
    p = vpd_["n"].sub(1, deepcopy=True)
    d = vpd_["n"].sub(2, deepcopy=True)
    d_n1 = vpd_["n-1"].sub(2, deepcopy=True)

    w_test.vector().zero()
    w_test.vector().axpy(1, d_tilde.vector())
    w_test.vector().axpy(-1, d_n1.vector())
    w_test.vector()[:] = w_test.vector()[:] / dt

    if counter%step ==0:
        u_file.write(v)
        d_file.write(d)
        dtilde_file.write(d_tilde)
        v_tilde_file.write(v_tilde)
        p_file.write(p)
        w_file.write(w_test)

    def F_(U):
        return (Identity(len(U)) + grad(U))

    def J_(U):
        return det(F_(U))

    def sigma_f_new(v, p, d, mu_f):
        return -p*Identity(len(v)) + mu_f*(grad(v)*inv(F_(d)) + inv(F_(d)).T*grad(v).T)

    def sigma_f(p, u, d, mu_f):
        return  -p*Identity(len(u)) +\
                mu_f*(grad(u)*inv(F_(d)) + inv(F_(d)).T*grad(u).T)

    #Det = project(J_(d), DVP.sub(0).collapse())
    #Det = project(J_(d), P)
    #Det_list.append((Det.vector().array()).min())

    Dr = -assemble((sigma_f_new(v, p, d, mu_f)*n)[0]*ds2(6))
    Li = -assemble((sigma_f_new(v, p, d, mu_f)*n)[1]*ds2(6))
    Dr += -assemble((sigma_f_new(v("+"), p("+"), d("+"), mu_f)*n("+"))[0]*dS2(5))
    Li += -assemble((sigma_f_new(v("+"), p("+"), d("+"), mu_f)*n("+"))[1]*dS2(5))

    Drag_list.append(Dr)
    Lift_list.append(Li)
    Time_list.append(t)

    dsx = d(coord)[0]
    dsy = d(coord)[1]
    dis_x.append(dsx)
    dis_y.append(dsy)
    if MPI.rank(mpi_comm_world()) == 0:
        print "LIFT = %g,  DRAG = %g" % (Li, Dr)
        print "dis_x/dis_y : %g %g "%(dsx,dsy)

    return {}


def post_process(path, T, dt, Det_list, dis_x, dis_y, Drag_list, Lift_list, Time_list,\
                mesh_name, args, simtime, v_deg, p_deg, d_deg, vpd_file, **semimp_namespace):
    theta = args.theta
    f_scheme = args.fluidvari
    s_scheme = args.solidvari
    e_scheme = args.extravari
    f = open(path+"/report.txt", 'w')
    f.write("""FSI3 EXPERIMENT
    T = %(T)g\ndt = %(dt)g\nv_deg = %(d_deg)g\nv_deg = %(v_deg)g\np_deg = %(p_deg)g\n
theta = %(theta)s\nf_vari = %(f_scheme)s\ns_vari = %(s_scheme)s\ne_vari = %(e_scheme)s\n time = %(simtime)g""" % vars())
    f.write("\nRuntime = %f" % fintime)
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
