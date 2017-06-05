from dolfin import *
import numpy as np

mesh_file = UnitSquareMesh(10, 10)
#mesh_file = refine(mesh_file)
#Parameters for each numerical case
common = {"mesh": mesh_file,
          "v_deg": 2,    #Velocity degree
          "p_deg": 1,    #Pressure degree
          "d_deg": 2,    #Deformation degree
          "T": 1E-4,          # End time
          "dt": 1E-5,       # Time step
          "rho_f": 1.0,    #
          "mu_f": 1.,
          "rho_s" : Constant(1.0),
          "mu_s" : Constant(1.0),
          "nu_s" : Constant(1.0)
     }

vars().update(common)
lamda_s = nu_s*2*mu_s/(1 - 2.*nu_s)
nu = mu_f/rho_f
#plot(mesh, interactive=True)


#Allboundaries = DomainBoundary()
Fluid_area = AutoSubDomain(lambda x, on_bnd: (x[0] <= 0.5) and on_bnd)
boundaries = FacetFunction("size_t", mesh_file)
boundaries.set_all(0)
DomainBoundary().mark(boundaries, 2)
Fluid_area.mark(boundaries, 1)

plot(boundaries,interactive=True)

ds = Measure("ds", subdomain_data = boundaries)
dS = Measure("dS", subdomain_data = boundaries)
n = FacetNormal(mesh_file)

Fluid_area = AutoSubDomain(lambda x: (x[0] <= 0.5))
domains = CellFunction("size_t", mesh_file)
domains.set_all(2)
Fluid_area.mark(domains, 1)
dx = Measure("dx", subdomain_data = domains)

#plot(domains,interactive = True)

dx_f = dx(1, subdomain_data = domains)
dx_s = dx(2, subdomain_data = domains)

def initiate(DVP, dvp_, mesh_file, t, rho_f, rho_s, mu_f, mu_s, nu, \
            dx_f, dx_s, F_fluid_linear, F_solid_linear, phi, psi, **monolithic):
    t_ = Constant(0)
    x = SpatialCoordinate(mesh_file)

    def sigma_f(p_, u_, mu_f):
        return -p_*Identity(2) + 2.*mu_f*sym(grad(u_))

    def F_(U):
    	return Identity(len(U)) + grad(U)

    def J_(U):
    	return det(F_(U))

    def E(U):
    	return 0.5*(F_(U).T*F_(U) - Identity(len(U)))

    def S(U,lamda_s,mu_s):
        I = Identity(len(U))
        return 2*mu_s*E(U) + lamda_s*tr(E(U))*I

    def Piola1(U,lamda_s,mu_s):
    	return F_(U)*S(U,lamda_s,mu_s)

    d_x = "0"
    d_y = "t_*(x[0] - 0.5)"
    u_x = "0"
    u_y = "(x[0] - 0.5)"
    p_c = "2"

    p_e = Expression(p_c, nu=nu, t_=0.0, degree=6)

    u_e = Expression((u_x,\
                    u_y), nu=nu, t_=0.0, degree=6)

    d_e = Expression((d_x,\
                    d_y), nu=nu, t_=0.0, degree=6)

    assign(dvp_["n-1"].sub(0), interpolate(d_e, DVP.sub(0).collapse()))
    assign(dvp_["n-1"].sub(1), interpolate(u_e, DVP.sub(1).collapse()))
    assign(dvp_["n-1"].sub(2), interpolate(p_e, DVP.sub(2).collapse()))

    #For a non-zero initial guess on first newtoniteration
    assign(dvp_["n"].sub(0), interpolate(d_e, DVP.sub(0).collapse()))
    assign(dvp_["n"].sub(1), interpolate(u_e, DVP.sub(1).collapse()))
    assign(dvp_["n"].sub(2), interpolate(p_e, DVP.sub(2).collapse()))

    t_ = Constant(0)
    d_x = 0
    d_y = t_*(x[0] - 0.5)
    u_x = 0
    u_y = (x[0] - 0.5)
    p_c = 2

    #exec("d_x = %s" % d_x) in locals(), globals()
    #exec("d_y = %s" % d_y) in locals(), globals()
    #exec("u_x = %s" % u_x) in locals(), globals()
    #exec("u_y = %s" % u_y) in locals(), globals()
    #exec("p_c = %s" % p_c) in locals(), globals()

    d_vec = as_vector([d_x, d_y])
    u_vec = as_vector([u_x, u_y])

    f_fluid = rho_f*diff(u_vec, t_) + rho_f*dot(grad(u_vec), u_vec - diff(d_vec, t_)) - div(sigma_f(p_c, u_vec, mu_f))
    #f_fluid = rho_f*diff(u_vec, t_) + rho_f*dot(u_vec - diff(d_vec, t_), grad(u_vec)) - div(sigma_f(p_c, u_vec, mu_f))

    f_solid = rho_s*diff(d_vec, t_) - div(Piola1(d_vec, lamda_s, mu_s))

    F_fluid_linear -= inner(Constant((10,0)), phi)*dx_f
    #F_fluid_linear -= inner(J_(d_vec)*f_fluid, phi)*dx_f
    F_solid_linear -= inner(f_solid, psi)*dx_s

    return dict(t_=t_, d_e=d_e, u_e=u_e, p_e=p_e)

def create_bcs(DVP, dvp_, u_e, p_e, d_e, boundaries, **semimp_namespace):
    #displacement conditions:
    d_bc    = DirichletBC(DVP.sub(0), d_e, "on_boundary")

    #Fluid velocity conditions
    u_bc  = DirichletBC(DVP.sub(1), u_e, "on_boundary")

    #Pressure Conditions
    p_bc = DirichletBC(DVP.sub(2), p_e, boundaries, 1)

    #Assemble boundary conditions
    bcs = [d_bc, u_bc, p_bc]

    return dict(bcs = bcs)

def pre_solve(t, t_, p_e, d_e, u_e, **semimp_namespace):
    t_.assign(t)
    p_e.t_ = t
    u_e.t_ = t
    d_e.t_ = t
    return {}


def after_solve(**semimp_namespace):


    return {}

def post_process(E_u, u_e, d_e, dvp_, mu_f, lamda_s, mu_s, mesh_file, boundaries, **semimp_namespace):
    def F_(U):
    	return (Identity(len(U)) + grad(U))

    def J_(U):
    	return det(F_(U))

    def E(U):
    	return 0.5*(F_(U).T*F_(U) - Identity(len(U)))

    def S(U,lamda_s,mu_s):
        I = Identity(len(U))
        return 2*mu_s*E(U) + lamda_s*tr(E(U))*I

    def Piola1(U,lamda_s,mu_s):
    	return F_(U)*S(U,lamda_s,mu_s)

    def sigma_f_new(v, p, d, mu_f):
    	return -p*Identity(len(v)) + mu_f*(grad(v)*inv(F_(d)) + inv(F_(d)).T*grad(v).T)

    d, v, p = dvp_["n"].split(True)

    F_Dr = -assemble((sigma_f_new(v("-"),p("-"),d("-"),mu_f)*n("-"))[0]*dS(5))
    F_Li = -assemble((sigma_f_new(v("-"),p("-"),d("-"),mu_f)*n("-"))[1]*dS(5))

    S_Dr = -assemble((Piola1(d("-"),lamda_s, mu_s)*n("-"))[0]*dS(5))
    S_Li = -assemble((Piola1(d("-"),lamda_s, mu_s)*n("-"))[0]*dS(5))


    #To evaluate error on fluid and structure
    submesh_fluid = SubMesh(mesh, boundaries, 1)
    submesh_struc = SubMesh(mesh, boundaries, 2)

    #E_d = errornorm(d_e, d_s, norm_type="l2", degree_rise=2, mesh=submesh_struc)
    E_u.append(errornorm(u_e, v_s, norm_type="l2", degree_rise=2, mesh=submesh_fluid))
    h_ = mesh_file.hmin()
    #print "E_u=%g, E_d=%g" % (E_u, E_d)
    return {}
