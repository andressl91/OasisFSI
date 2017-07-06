from dolfin import *
import numpy as np

mesh_file = Mesh("Mesh/fluid_new.xml")
#mesh_file = refine(mesh_file)
#Parameters for each numerical case
common = {"mesh": mesh_file,
          "v_deg": 2,    #Velocity degree
          "p_deg": 1,    #Pressure degree
          "d_deg": 2,    #Deformation degree
          "T": 0.5,          # End time
          "dt": 0.5,       # Time step
          "rho_f": 1.0E3,    #
          "mu_f": 1.,
          "rho_s" : Constant(10E6),
          "mu_s" : Constant(10E12),
          "nu_s" : Constant(10E10),
          "Um" : 0.2,
          "D" : 0.1,
          "H" : 0.41,
          "L" : 2.5,
          "step": 1 #Which timestep to store solution
     }

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

def initiate(v_deg, d_deg, p_deg, dt, theta, dw_, vp_, args, mesh_name, refi, **semimp_namespace):
    exva = args.extravari
    extype = args.extype
    bitype = args.bitype
    if args.extravari == "alfa":
        path = "FSI_fresh_results/FSI-1/%(exva)s_%(extype)s/dt-%(dt)g_theta-%(theta)g/%(mesh_name)s_refine_%(refi)d_v_deg_%(v_deg)s_d_deg_%(d_deg)s_p_deg_%(p_deg)s" % vars()
    if args.extravari == "biharmonic" or args.extravari == "laplace" or args.extravari == "elastic" or args.extravari == "biharmonic2":
        path = "FSI_fresh_results/FSI-1/%(exva)s_%(bitype)s/dt-%(dt)g_theta-%(theta)g/%(mesh_name)s_refine_%(refi)d_v_deg_%(v_deg)s_d_deg_%(d_deg)s_p_deg_%(p_deg)s" % vars()

    u_file = XDMFFile(mpi_comm_world(), path + "/velocity.xdmf")
    d_file = XDMFFile(mpi_comm_world(), path + "/d.xdmf")
    p_file = XDMFFile(mpi_comm_world(), path + "/pressure.xdmf")
    for tmp_t in [u_file, d_file, p_file]:
        tmp_t.parameters["flush_output"] = True
        tmp_t.parameters["multi_file"] = 0
        tmp_t.parameters["rewrite_function_mesh"] = False
    #d = dvp_["n-1"].sub(0, deepcopy=True)
    d = dw_["n"].sub(0, deepcopy=True)
    v = vp_["n"].sub(1, deepcopy=True)
    p = vp_["n"].sub(0, deepcopy=True)
    p_file.write(p)
    d_file.write(d)
    u_file.write(v)
    #u_file << v
    #d_file << d

    return dict(u_file=u_file, d_file=d_file, p_file=p_file, path=path)

def create_bcs(dw_, d_, DW, VP, args, k, Um, H, boundaries, inlet, **semimp_namespace):
    print "Create bcs"

    wm_inlet  = DirichletBC(DW.sub(0), ((0.0, 0.0)), boundaries, 3)
    wm_outlet = DirichletBC(DW.sub(0), ((0.0, 0.0)), boundaries, 4)
    wm_wall   = DirichletBC(DW.sub(0), ((0.0, 0.0)), boundaries, 2)
    wm_circ   = DirichletBC(DW.sub(0), ((0.0, 0.0)), boundaries, 6)
    wm_bar    = DirichletBC(DW.sub(0).collapse(), d_["tilde"], boundaries, 5)
    bcs_w = [wm_wall, wm_inlet, wm_outlet, wm_circ, wm_bar]

    # Fluid tentative bcs
    #d_tilde = dw_["tilde"].sub(0)
    #d_n1 = dw_["n-1"].sub(0)
    d_tilde = d_["tilde"]
    d_n1 = d_["n-1"]
    w_bar = 1./k*(d_tilde - d_n1)

    u_inlet_t  = DirichletBC(VP.sub(0), inlet, boundaries, 3)
    u_wall_t   = DirichletBC(VP.sub(0), ((0.0, 0.0)), boundaries, 2)
    u_circ_t   = DirichletBC(VP.sub(0), ((0.0, 0.0)), boundaries, 6) #No slip on geometry in fluid
    u_bar_t    = DirichletBC(VP.sub(0).collapse(), w_bar, boundaries, 5) #No slip on geometry in fluid

    bcs_tent = [u_wall_t, u_inlet_t, u_circ_t, u_bar_t]

    #Fluid correction bcs
    u_inlet  = DirichletBC(VP.sub(0), inlet, boundaries, 3)
    u_wall   = DirichletBC(VP.sub(0), ((0.0, 0.0)), boundaries, 2)
    u_circ   = DirichletBC(VP.sub(0), ((0.0, 0.0)), boundaries, 6) #No slip on geometry in fluid

    #p_outlet  = DirichletBC(VP.sub(1), -60, boundaries, 4)
    p_outlet  = DirichletBC(VP.sub(1), (0.0), boundaries, 4)

    #Assemble boundary conditions
    bcs_corr = [u_wall, u_inlet, u_circ, \
                p_outlet]

    bcs_solid = []
    #if DVP.num_sub_spaces() == 4:
    if args.bitype == "bc1":
        u_barwall= DirichletBC(DW.sub(1), ((0.0, 0.0)), boundaries, 7)
        d_barwall = DirichletBC(DW.sub(0), ((0.0, 0.0)), boundaries, 7) #No slip on geometry in fluid
        for i in [d_barwall, u_barwall]:
            bcs_solid.append(i)

    return dict(bcs_tent=bcs_tent, bcs_w=bcs_w, bcs_corr=bcs_corr, \
                bcs_solid=bcs_solid)

def pre_solve(t, inlet, **semimp_namespace):
    if t < 2:
        inlet.t = t
    else:
        inlet.t = 2

    return dict(inlet = inlet)

def after_solve(t, dvp_, n,coord,dis_x,dis_y,Drag_list,Lift_list, **semimp_namespace):
    #d = dvp_["n"].sub(0, deepcopy=True)
    #v = dvp_["n"].sub(1, deepcopy=True)
    #p = dvp_["n"].sub(2, deepcopy=True)
    d, v, p = dvp_["n"].split(True)

    def F_(U):
    	return (Identity(len(U)) + grad(U))

    def J_(U):
    	return det(F_(U))

    def sigma_f_new(v, p, d, mu_f):
    	return -p*Identity(len(v)) + mu_f*(grad(v)*inv(F_(d)) + inv(F_(d)).T*grad(v).T)

    #Fx = -assemble((sigma_f_new(v, p, d, mu_f)*n)[0]*ds(6))
    #Fy = -assemble((sigma_f_new(v, p, d, mu_f)*n)[1]*ds(6))
    #Fx += -assemble(((-p("-")*Identity(len(v)) + mu_f*(grad(v)("-")*inv(F_(d("-"))) + inv(F_(d("-"))).T*grad(v)("-").T))*n('-'))[0]*dS(5))
    #Fy += -assemble(((-p("-")*Identity(len(v)) + mu_f*(grad(v)("-")*inv(F_(d("-"))) + inv(F_(d("-"))).T*grad(v)("-").T))*n('-'))[1]*dS(5))
    Dr = -assemble((sigma_f_new(v,p,d,mu_f)*n)[0]*ds(6))
    Li = -assemble((sigma_f_new(v,p,d,mu_f)*n)[1]*ds(6))
    Dr += -assemble((sigma_f_new(v("-"),p("-"),d("-"),mu_f)*n("-"))[0]*dS(5))
    Li += -assemble((sigma_f_new(v("-"),p("-"),d("-"),mu_f)*n("-"))[1]*dS(5))
    Drag_list.append(Dr)
    Lift_list.append(Li)

    print "LIFT = %g,  DRAG = %g" % (Li, Dr)

    dsx = d(coord)[0]
    dsy = d(coord)[1]
    dis_x.append(dsx)
    dis_y.append(dsy)
    print "dis_x/dis_y : %g %g "%(dsx,dsy)

    return {}

def post_process(T,dt,dis_x,dis_y, Drag_list,Lift_list,**semimp_namespace):
    return {}
