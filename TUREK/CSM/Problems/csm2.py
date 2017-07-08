from dolfin import *
import numpy as np
import matplotlib.pyplot as plt
import sys

mesh_name = "base0"
mesh_file = Mesh("Mesh/" + mesh_name +".xml")
#mesh_file = refine(mesh_file)
#Parameters for each numerical case
common = {"mesh": mesh_file,
          "v_deg": 2,    #Velocity degree
          "d_deg": 2,    #Deformation degree
          "T": 10.0,          # End time
          "dt": 0.1,       # Time step
          "rho_s" : Constant(1.0E3),
          "mu_s" : Constant(2.0E6),
          "nu_s" : Constant(0.4),
          "D" : 0.1,
          "H" : 0.41,
          "L" : 2.5,
    	  "step" : 1,
          "checkpoint": False
          }
 #"checkpoint": "./FSI_fresh_checkpoints/CSM-2/P-2/dt-0.05/dvFile.h5"
g = 2
vars().update(common)
lamda_s = nu_s*2*mu_s/(1 - 2.*nu_s)
#plot(mesh, interactive=True)
for coord in mesh.coordinates():
    if coord[0]==0.35 and (0.00999<=coord[1]<=0.1001): # to get the point [0.2,0.6] end of bar
        print coord
        break
# BOUNDARIES

#NOS = AutoSubDomain(lambda x: "on_boundary" and( near(x[1],0) or near(x[1], 0.41)))
Left = AutoSubDomain(lambda x: "on_boundary" and near(x[0],0))

Allboundaries = DomainBoundary()

boundaries = FacetFunction("size_t",mesh_file)
boundaries.set_all(0)
Allboundaries.mark(boundaries, 1)
Left.mark(boundaries, 2)

#plot(boundaries, interactive=True)


ds = Measure("ds", subdomain_data = boundaries)
dS = Measure("dS", subdomain_data = boundaries)
n = FacetNormal(mesh_file)

Time_list = []
dis_x = []
dis_y = []

dv_file=HDF5File(mpi_comm_world(), "FSI_fresh_checkpoints/CSM-2/P-"+str(v_deg)+"/dt-"+str(dt)+"/dummy.h5", "w")

def initiate(t, T, args, mesh_file, mesh_name, \
            d_deg, v_deg, dt, dv_, Time_list,**semimp_namespace):

    theta = args.theta
    path = "FSI_fresh_results/CSM-2/dt-%(dt)g_theta-%(theta)g/%(mesh_name)s_d_deg_%(d_deg)s_v_deg_%(v_deg)s" % vars()
    dummy_file=HDF5File(mpi_comm_world(), path + "/dummy.h5", "w")
    d = dv_["n"].sub(0, deepcopy=True)
    Time_list.append(t)
    dsx = d(coord)[0]
    dsy = d(coord)[1]
    dis_x.append(dsx)
    dis_y.append(dsy)

    return dict(path=path)

def create_bcs(VV, dv_,boundaries,  **semimp_namespace):

    #Fluid velocity conditions
    d_left  = DirichletBC(VV.sub(0), ((0.0, 0.0)), boundaries, 2)
    v_left  = DirichletBC(VV.sub(1), ((0.0, 0.0)), boundaries, 2)

    #Assemble boundary conditions
    bcs = [d_left, v_left]

    return dict(bcs = bcs)


def pre_solve(**semimp_namespace):
    return {}


def after_solve(t, path, dv_, n,coord,dis_x,dis_y, \
                counter, **semimp_namespace):

    d = dv_["n"].sub(0, deepcopy=True)
    v = dv_["n"].sub(1, deepcopy=True)
    #if counter%step ==0:
#        u_file << v
#        d_file << d
        #p_file << p
        #p_file.write(p)
        #d_file.write(d)
        #u_file.write(v)
        #dv_file << dv_["n"]
        #dv_file.write(dv_["n"], "dv%g"%t)

    Time_list.append(t)
    dsx = d(coord)[0]
    dsy = d(coord)[1]
    dis_x.append(dsx)
    dis_y.append(dsy)
    if MPI.rank(mpi_comm_world()) == 0:
        print "dis_x/dis_y : %g %g "%(dsx,dsy)

    return {}

def post_process(path,T,dt, dis_x,dis_y, Time_list,\
                    args, v_deg, d_deg, **semimp_namespace):

    theta = args.theta

    if MPI.rank(mpi_comm_world()) == 0:

        f = open(path+"/report.txt", 'w')
        f.write("""CSM-2 EXPERIMENT
        T = %(T)g\ndt = %(dt)g\nv_deg = %(d_deg)g\nv_deg = %(v_deg)g\n
        theta = %(theta)s\n""" % vars())
        #f.write("""Runtime = %f """ % fintime)
        f.close()

        np.savetxt(path + '/Time.txt', Time_list, delimiter=',')
        np.savetxt(path + '/dis_x.txt', dis_x, delimiter=',')
        np.savetxt(path + '/dis_y.txt', dis_y, delimiter=',')

    return {}
