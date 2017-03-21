from dolfin import *

#Parameters for each numerical case
con_space = {
          "v_deg": 2,    #Velocity degree
          "p_deg": 1,    #Pressure degree
          "T": 2E-4,          # End time
          "Dt": [1e-5],       # Time step
          "N": [4, 8 ,12 ,16],
          "rho_f": 1,    #
          "mu_f": 1.,
     }

con_time = {"mesh": mesh,
          "v_deg": 2,    #Velocity degree
          "p_deg": 1,    #Pressure degree
          "T": 2E-1,          # End time
          "Dt": [5E-2, 4E-2, 2E-2],       # Time step
          "N": [50],
          "rho_f": 1.,    #
          "mu_f": 1.,
     }

vars().update(con_time)

#Machine Precicion
#u_x = "x[1]"
#u_y = "x[0]"
#p_c = "2"

#MMS
u_x = "cos(x[0])*sin(x[1])*cos(t_)"
u_y = "-sin(x[0])*cos(x[1])*cos(t_)"
p_c = "sin(x[0])*cos(x[1])*cos(t_)"

def sigma_f(p, u, mu):
    return -p*Identity(len(u)) + 2.*mu*sym(grad(u))

def pre_solve(t, t_, u_e, p_e, **semimp_namespace):
    u_e.t_ = t
    p_e.t_ = t
    t_.assign(t)
    #return {}
    return dict(u_e = u_e, p_e = p_e, t_ = t_)

def sourceterm(mesh, u_x, u_y, p_c, dt, F, rho_f, mu_f, v, dx, **semimp_namespace):
    x = SpatialCoordinate(mesh)
    t_ = Constant(dt)

    exec("u_x = %s" % u_x)
    exec("u_y = %s" % u_y)
    exec("p_c = %s" % p_c)
    u_vec = as_vector([u_x, u_y])

    #f = Constant((0, 0))

    f = rho_f*diff(u_vec, t_) + rho_f*dot(u_vec, grad(u_vec)) \
    - div(sigma_f(p_c, u_vec, mu_f))
    F -= inner(f, v)*dx
    return dict(F = F, x = x, t_ = t_)

def create_bcs(VQ, u_x, u_y, p_c, nu, rho_f, **semimp_namespace):
    bcs = []
    u_e = Expression((u_x,
                      u_y,
                     ), t_ = 0, nu = nu, degree = 5)

    p_e = Expression(p_c, t_ = 0, rho_f = rho_f, nu = nu, degree = 5)

    bc_u = DirichletBC(VQ.sub(0), u_e, "on_boundary")
    bc_p = DirichletBC(VQ.sub(1), p_e, "on_boundary")
    bcs = [bc_u, bc_p]

    return dict(bcs = bcs, u_e = u_e, p_e = p_e)

def post_process(V, u_sol, p_sol, E_u, E_p, u_e, p_e, **semimp_namespace):
    E_u.append(errornorm(u_e, u_sol, norm_type = "l2", degree_rise = 2))
    E_p.append(errornorm(p_e, p_sol, norm_type = "l2", degree_rise = 2))
    return dict(E_u = E_u, E_p = E_p)
