from dolfin import *

#Parameters for each numerical case
con_space = {
          "v_deg": 2,    #Velocity degree
          "p_deg": 1,    #Pressure degree
          "T": 2E-5,          # End time
          "Dt": [1e-6],       # Time step
          "N": [20, 25, 30],
          "rho_f": 1,    #
          "mu_f": 1.,
     }

con_time = {
          "v_deg": 2,    #Velocity degree
          "p_deg": 1,    #Pressure degree
          "T": 2E-2,          # End time
          "Dt": [5E-3, 4E-3, 2E-3],       # Time step
          "N": [40],
          "rho_f": 10.,    #
          "mu_f": 1.,
     }

vars().update(con_space)

def initiate(rho_f, mu_f, nu, **semimp_namespace):
    u_e = Expression(('-sin(pi*x[1])*cos(pi*x[0])*exp(-2.*pi*pi*nu*t)',
                  'sin(pi*x[0])*cos(pi*x[1])*exp(-2.*pi*pi*nu*t)'), t=0, nu=nu, degree=4)
    #SHOULD RHO BE IN p_e
    p_e = Expression(('-(cos(2*pi*x[0])+cos(2*pi*x[1]))*exp(-4.*pi*pi*nu*t)/4.'), t=0, rho=rho_f, nu=nu, degree=4)
    return dict(u_e = u_e, p_e = p_e)


def pre_solve(t, u_e, p_e, **semimp_namespace):
    u_e.t = t
    p_e.t = t
    return {}
    #return dict(u_e = u_e, p_e = p_e)

def sourceterm(mesh, **semimp_namespace):
    return {}

def create_bcs(**semimp_namespace):
    bcs_u = []; bcs_p = []
    return dict(bcs_u = bcs_u, bcs_p = bcs_p)

def post_process(V, Q, u_, p_, E_u, E_p, u_e, p_e, **semimp_namespace):
    E_u.append(errornorm(u_e, u_["sol"], norm_type = "l2", degree_rise = 3))
    E_p.append(errornorm(p_e, p_["sol"], norm_type = "l2", degree_rise = 3))

    return dict(E_u = E_u, E_p = E_p)
