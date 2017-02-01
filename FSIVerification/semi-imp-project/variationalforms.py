from dolfin import *

#sa = 2

I = Identity(2)
def eps(u):
    return 0.5*(grad(u) + grad(u).T)

def F_(U):
    return I + grad(U)

def S(U, mu_s, lamda_s):
	return (2*mu_s*E(U) + lamda_s*tr(E(U))*I)

def E(U):
	return 0.5*(F_(U).T*F_(U)-I)

def J_(U):
	return det(F_(U))

def P1(U, mu_s, lamda_s):
	return F_(U)*S(U, mu_s, lamda_s)

def sigma_f(v, p, mu_f):
	return 2*mu_f*sym(grad(v)) - p*Identity(2)

def sigma_f_hat(v, p, u, mu_f):
	return J_(u)*sigma_f(v, p, mu_f)*inv(F_(u)).T

def step0(u_, d0, v0, v_1, k ,phi, dx):
    F_expo = inner(u_ - d0 - k*(3./2*v0 - 1./2*v_1), phi)*dx
    return F_expo

def step1(w, d_tilde, d0, k, phi, dx_s, dx_f):
    F_smooth = inner(w - 1./k*(d_tilde - d0), phi)*dx_s\
             + inner(grad(w), grad(phi))*dx_f

    return F_smooth

def step2(d_, w_next, u_, u0, u0_tilde, phi, dx_f, mu_f, rho_f, k):
    F_tent = (rho_f/k)*inner(J_(d_)*(u_ - u0), phi)*dx_f \
            + rho_f*inner(J_(d_)*inv(F_(d_))*grad(u_)*(u0_tilde - w_next), phi)*dx_f  \
            + inner(2.*mu_f*J_(d_)*eps(u_)*inv(F_(d_)).T, eps(phi))*dx_f \
            + inner(u_('-') - w_next('-'), phi('-'))*dS(5)
            #proposed in winterschoolfsi
            #+ inner( 2.*mu_f*J_(d_move)*(grad(u_)*inv(F_(d_move)) + inv(F_(d_move)).T*grad(u_).T )*inv(F_()).T, eps(phi) )*dx_f \

    return F_tent


# Pressure update
def step3_1(d, d0, u, u_tent, p, psi, eta, n, dx_f, rho_f, k, dS):
    F_press_upt = (rho_f/k)*inner(J_(d)*(u - u_tent), psi)*dx_f \
    - inner(J_(d)*p*inv(F_(d)).T, grad(psi))*dx_f \
    + inner(div(J_(d)*inv(F_(d).T)*u), eta)*dx_f \
    + inner(dot(u('-'),n('-')), eta('-'))*dS(5) \
    - 1./k*inner(dot(d('-') - d0('-'), n('-')), eta('-'))*dS(5)
    #+ dot(dot(u('-'), n('-')) - 1./k*dot(d('-') - d0('-'), n('-')), psi('-'))*dS(5)
    return F_press_upt

def step3_2(v, v0, d, d0, u_s, p_s, k, mu_f, mu_s, rho_s, lamda_s, n, dx_s, alfa, beta, dS):
    Solid_v = rho_s/k*inner(v - v0, alfa)*dx_s + 0.5*inner(P1(d, mu_s, lamda_s) + P1(d0, mu_s, lamda_s), eps(alfa))*dx_s
    Solid_d = 1.0/k*inner(d - d0, beta)*dx_s - 0.5*inner(v + v0, beta)*dx_s
    #Solid_d = delta*(1.0/k*inner(d - d0, beta)*dx_s - 0.5*inner(v + v0, beta)*dx_s)
    Solid_dynamic = inner(P1(d('+'), mu_s, lamda_s)*n('+'), beta('+'))*dS(5) \
                  - inner(sigma_f_hat(u_s('+'), p_s('+'), d('+'), mu_f)*n('+'), beta('+'))*dS(5)
    F_s = Solid_v + Solid_d + Solid_dynamic
    return F_s
