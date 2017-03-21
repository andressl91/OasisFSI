from dolfin import *
from mappings import *
from parser import *
from Hron_Turek import *
def var_form(d,d0,d1,d2,d_,df,df0,phi,VQ,V1,u,u_,u0,traction_,gamma,psi,p,p_,beta,delta,Q,rho_s,rho_f,k,dx_s,dx_f,mu_f):

    # Lifting operator

    f = Function(V1)
    #f, _ = f_.split()
    F_Ext = inner(grad(d), grad(psi))*dx_f - inner(f, psi)*dx_f #- inner(grad(d)*n, psi)*ds

    # Structure variational form
    #F_structure = (rho_s/(k*k))*inner(d-2*d0+d1, psi)*dx_s
    #F_structure = (rho_s/k)*inner(u-u0,phi)*dx_s
    F_structure = inner(sigma_dev(d), grad(psi))*dx_s #+ ??alpha*(rho_s/k)*(0.5*(d-d1))*dx_s??
    #F_structure += ((1.0/k)*inner(d-d0,psi)*dx_s - inner(0.5*(u_+u0), psi)*dx_s)
    #F_structure += delta*((1.0/k)*inner(d("-")-d0("-"),psi("-"))*dS(5) - inner(u_("-"), psi("-"))*dS(5))
    F_structure += inner(sigma_dev(d("-"))*n("-"), psi("-"))*dS(5)
    F_structure += inner(J_(d0("-"))*sigma_f_new(u_("-"),p_("-"),d0("-"),mu_f)*inv(F_(df("-"))).T*n("-"), psi("-"))*dS(5)

    #F_structure += inner(traction_, psi("-"))*dS(5) # Idea from IBCS
    #F_structure -= inner(Constant((0,-4*rho_s)), psi)*dx_s # Gravita


    #F_structure += inner(J_(d("-"))*sigma_f(u_("-"),p_("-"))*inv(F_(d("-"))).T*n("-"), psi("-"))*dS(5)
    #F_structure = inner(sigma_f_new(uf("-"),pf("-"),d("-"))*n("-"), psi("-"))*dS(5)

    # Fluid variational form
    F_fluid = (rho_f/k)*inner(J_(df)*(u - u0), phi)*dx_f
    F_fluid += rho_f*inner(J_(df)*grad(u)*inv(F_(df))*(u0 - ((df-df0)/k)), phi)*dx_f
    F_fluid += inner(J_(df)*sigma_f_new(u,p,df,mu_f)*inv(F_(df)).T, grad(phi))*dx_f
    F_fluid -= inner(div(J_(df)*inv(F_(df)).T*u), gamma)*dx_f
    F_fluid += inner(J_(df("-"))*sigma_f_new(u("-"),p("-"),df("-"),mu_f)*inv(F_(df("-"))).T*n("-"), phi("-"))*dS(5)
    F_fluid += inner(sigma_dev(d_("-"))*n("-"), phi("-"))*dS(5)
    #F_fluid -= beta*h*h*inner(J_(df)*inv(F_(df).T)*grad(p), grad(gamma))*dx_f
    F_fluid += (rho_s/k)*delta*((1.0/k)*inner(d0("-")-d1("-"),phi("-"))*dS(5) - inner(u("-"), phi("-"))*dS(5))

    af = lhs(F_fluid)
    bf = rhs(F_fluid)

    a_s = lhs(F_structure)
    b_s = rhs(F_structure)

    adf = lhs(F_Ext)
    Ldf = rhs(F_Ext)

    return af, bf, a_s, b_s, adf, Ldf
