from dolfin import *
import numpy as np
import os

mesh_name = "base0"
mesh_file = Mesh("Mesh/" + mesh_name +".xml")
#mesh_file = refine(mesh_file)
#Parameters for each numerical case
common = {"mesh": mesh_file,
          "v_deg": 2,    #Velocity degree
          "p_deg": 1,    #Pressure degree
          "T": 0.5,          # End time
          "dt": 0.5,       # Time step
          "rho_f": 1.0E3,    #
          "mu_f": 1.,
          "Um" : 1.0,
          "D" : 0.1,
          "H" : 0.41,
          "L" : 2.5,
          "step": 0 #Which timestep to store solution
     }

vars().update(common)

Inlet = AutoSubDomain(lambda x: "on_boundary" and near(x[0],0))
Outlet = AutoSubDomain(lambda x: "on_boundary" and (near(x[0],2.5)))
Wall =  AutoSubDomain(lambda x: "on_boundary" and (near(x[1], 0.41) or near(x[1], 0)))

Allboundaries = DomainBoundary()

boundaries = FacetFunction("size_t",mesh_file)
boundaries.set_all(0)
Allboundaries.mark(boundaries, 1) #Geometry
Wall.mark(boundaries, 2)
Inlet.mark(boundaries, 3)
Outlet.mark(boundaries, 4)


#plot(boundaries,interactive=True)

ds = Measure("ds", subdomain_data = boundaries)
n = FacetNormal(mesh_file)

Drag_list = []
Lift_list = []
Time_list = []
#Fluid properties
class Inlet(Expression):
    def __init__(self, Um):
        self.Um = Um
    	self.t = 0
    def eval(self,value,x):
    	value[0] = 0.5*(1-np.cos(self.t*np.pi/2))*1.5*self.Um*x[1]*(H-x[1])/((H/2.0)**2)
    	value[1] = 0
    def value_shape(self):
    	return (2,)

def initiate(args, dt, theta, v_deg, p_deg, mesh_name, **monolithic):
    theta = args.theta
    path = "FSI_fresh_results/CFD-2/dt-%(dt)g_theta-%(theta)g/%(mesh_name)s_v_deg_%(v_deg)s_p_deg_%(p_deg)s" % vars()
    return dict(path=path)

def create_bcs(VP, vp_, n, k, Um, H, boundaries, Inlet, **semimp_namespace):
    inlet = Inlet(Um)

    u_inlet = DirichletBC(VP.sub(0), inlet, boundaries, 3)
    nos_geo = DirichletBC(VP.sub(0), ((0, 0)), boundaries, 1)
    nos_wall = DirichletBC(VP.sub(0), ((0, 0)), boundaries, 2)

    p_out = DirichletBC(VP.sub(1), 0, boundaries, 4)

    bcs = [u_inlet, nos_geo, nos_wall, p_out]

    return dict(bcs = bcs, inlet = inlet)

def pre_solve(t, inlet, **semimp_namespace):
    if t < 2:
        inlet.t = t
    else:
        inlet.t = 2

    return dict(inlet = inlet)

def after_solve(t, vp_, n, Drag_list,Lift_list, **semimp_namespace):
    v, p = vp_["n"].split(True)

    def sigma_f(v, p, mu_f):
    	return -p*Identity(len(v)) + mu_f*(grad(v) + grad(v).T)


    Dr = -assemble((sigma_f(v,p,mu_f)*n)[0]*ds(1))
    Li = -assemble((sigma_f(v,p,mu_f)*n)[1]*ds(1))
    Drag_list.append(Dr)
    Lift_list.append(Li)
    Time_list.append(t)

    print "LIFT = %g,  DRAG = %g" % (Li, Dr)

    return {}

def post_process(path, T, dt, Drag_list,Lift_list, Time_list,\
                 simtime, args, v_deg, p_deg, **semimp_namespace):
    #dvp_file.close()
    #time_list = np.linspace(0,T,T/dt+1)
    theta = args.theta

    if MPI.rank(mpi_comm_world()) == 0:
        print "IN POSTPRO", path
        if not os.path.exists(path):
            os.makedirs(path)
        f = open(path+"/report.txt", 'w+')
        f.write("""CFD2 EXPERIMENT
        T = %(T)g\ndt = %(dt)g\nv_deg = %(v_deg)g\np_deg = %(p_deg)g\n
        theta = %(theta)s\n time = %(simtime)g""" % vars())
        #f.write("""Runtime = %f """ % fintime)
        f.close()

        np.savetxt(path + '/Lift.txt', Lift_list, delimiter=',')
        np.savetxt(path + '/Drag.txt', Drag_list, delimiter=',')
        np.savetxt(path + '/Time.txt', Time_list, delimiter=',')

    return {}
