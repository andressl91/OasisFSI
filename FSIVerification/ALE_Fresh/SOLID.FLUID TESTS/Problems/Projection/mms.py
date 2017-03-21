from dolfin import *

def sigma_f(p_, u_, mu_f):
    return -p_*Identity(2) + 2.*mu_f*sym(grad(u_))

#Parameters for each numerical case
con_space = {
          "v_deg": 2,    #Velocity degree
          "p_deg": 1,    #Pressure degree
          "T": 2E-5,          # End time
          "Dt": [1e-6],       # Time step
          "N": [4, 8 ,12 ,16],
          "rho_f": 1,    #
          "mu_f": 1.,
     }

con_time = {
          "v_deg": 2,    #Velocity degree
          "p_deg": 1,    #Pressure degree
          "T": 2E-2,          # End time
          "Dt": [5E-3, 4E-3, 2E-3],       # Time step
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

def pre_solve(t, t_, u_e, p_e, **semimp_namespace):
    u_e.t_ = t
    p_e.t_ = t
    t_.assign(t)
    return {}
    #return dict(u_e = u_e, p_e = p_e, t_ = t_)

def init(V, Q, u_, p_, rho_f, nu, n_, **semimp_namespace):
    u_e = Expression((u_x,
                      u_y,
                     ), t_ = 0, nu = nu, degree = 5)

    p_e = Expression(p_c, t_ = 0, rho_f = rho_f, nu = nu, degree = 5)

    return dict(u_e = u_e, p_e = p_e)

def sourceterm(mesh, u_x, u_y, p_c, dt, Fu_tent, rho_f, mu_f, v, dx, **semimp_namespace):
    x = SpatialCoordinate(mesh)
    t_ = Constant(dt)

    exec("u_x = %s" % u_x)
    exec("u_y = %s" % u_y)
    exec("p_c = %s" % p_c)
    u_vec = as_vector([u_x, u_y])

    #f = Constant((0, 0))
    f = diff(u_vec, t_) + dot(u_vec, grad(u_vec)) \
    - 1./rho_f*div(sigma_f(p_c, u_vec, mu_f))

    #f = diff(u_vec, t_) + dot(u_vec, grad(u_vec)) \
    #+ 1./rho_f*div(p_c*Identity(2)) - 2.*mu_f*div(sym(grad(u_vec)))

    Fu_tent -= inner(f, v)*dx
    return dict(Fu_tent = Fu_tent, x = x, t_ = t_)

def create_bcs(V, Q, u_e, p_e, **semimp_namespace):

    bcs_u = [DirichletBC(V, u_e, "on_boundary")]
    bcs_p = [DirichletBC(Q, p_e, "on_boundary")]

    return dict(bcs_u = bcs_u, bcs_p = bcs_p,u_e = u_e, p_e = p_e)

def post_process(u_sol, p_sol, E_u, E_p, u_e, p_e, **semimp_namespace):
    #ue = interpolate(ue, VV[ui])
    #uen = norm(ue.vector())
    #ue.vector().axpy(-1, q_[ui].vector())
    #final_error[i] = norm(ue.vector())/uen

    E_u.append(errornorm(u_e, u_sol, norm_type = "l2", degree_rise = 2))
    E_p.append(errornorm(p_e, p_sol, norm_type = "l2", degree_rise = 2))
    return dict(E_u = E_u, E_p = E_p)
