from dolfin import *
from mappings import *
from parser import *
from Hron_Turek import *
def var_form(d,d0,d1,d2,d_,df,df0,phi,VQ,V1,u,u_,u0,traction_,gamma,psi,p,p_,beta,delta,Q,rho_s,rho_f,k,dx_s,dx_f,XI):
    Mass_s_rhs = assemble((rho_s/(k*k))*inner(-2*d0+d1, psi)*dx_s) #solid mass time
    Mass_s_lhs = assemble((rho_s/(k*k))*inner(d, psi)*dx_s)

    Mass_s_b = assemble(inner(df("-"), phi("-"))*dS(5))

    #Mass_s_b_lhs = assemble((rho_s/k)*inner(u("-"), phi("-"))*dS(5))
    # TODO: Change d0 in V2 to VQ?
    Mass_s_b_rhs = assemble((rho_s/k)*inner((2*((d0("-")-d1("-"))/k) - ((d1("-") - d2("-"))/k)), phi("-"))*dS(5))

    ones_d = Function(V1)
    ones_u = Function(VQ)
    ones_d.vector()[:] = 1.
    ones_u.vector()[:] = 1.
    Mass_s_rhs_L = Mass_s_rhs*ones_d.vector() #Mass_time structure matrix lumped lhs
    Mass_s_lhs_L = Mass_s_lhs*ones_d.vector() #Mass_time structure matrix lumped rhs

    Mass_s_b_L = Mass_s_b*ones_u.vector() #Mass structure matrix lumped
    #Mass_s_and_lhs = Mass_s_b_L*Mass_s_b_lhs
    Mass_s_and_rhs = Mass_s_b_L*Mass_s_b_rhs


    mass_form = inner(u,phi)*dx
    M_lumped = assemble(mass_form)
    M_lumped.zero()
    M_lumped.set_diagonal(Mass_s_b_L)
    mass_time_form = inner(d,psi)*dx
    M_time_lumped_lhs = assemble(mass_time_form)
    M_time_lumped_lhs.zero()
    M_time_lumped_lhs.set_diagonal(Mass_s_lhs_L)
    #print type(M_time_lumped_lhs)
    M_time_lumped_rhs = assemble(mass_time_form)
    #print type(M_time_lumped_rhs)
    M_time_lumped_rhs.zero()
    M_time_lumped_rhs.set_diagonal(Mass_s_rhs_L)



    # Lifting operator

    f = Function(V1)
    #f, _ = f_.split()
    F_Ext = inner(grad(d), grad(XI))*dx_f - inner(f, XI)*dx_f #- inner(grad(d)*n, psi)*ds

    # Structure variational form

    F_structure = inner(sigma_dev(d), grad(psi))*dx_s #+ ??alpha*(rho_s/k)*(0.5*(d-d1))*dx_s??
    F_structure += delta*((1.0/k)*inner(d-d0,psi)*dx_s - inner(u_, psi)*dx_s)
    #F_structure += (1.0/k)*inner(d("-")-d0("-"),psi("-"))*dS(5) - inner(u_("-"), psi("-"))*dS(5)
    F_structure += inner(sigma_dev(d("-"))*n("-"), psi("-"))*dS(5)
    F_structure += inner(J_(d0("-"))*sigma_f_new(u_("-"),p_("-"),d0("-"))*inv(F_(df("-"))).T*n("-"), psi("-"))*dS(5)

    #F_structure += inner(traction_, psi("-"))*dS(5) # Idea from IBCS
    #F_structure -= inner(Constant((0,-4*rho_s)), psi)*dx_s # Gravita


    #F_structure += inner(J_(d("-"))*sigma_f(u_("-"),p_("-"))*inv(F_(d("-"))).T*n("-"), psi("-"))*dS(5)
    #F_structure = inner(sigma_f_new(uf("-"),pf("-"),d("-"))*n("-"), psi("-"))*dS(5)

    # Fluid variational form
    F_fluid = (rho_f/k)*inner(J_(df)*(u - u0), phi)*dx_f
    F_fluid += rho_f*inner(J_(df)*grad(u)*inv(F_(df))*(u0 - ((df-df0)/k)), phi)*dx_f
    F_fluid += (1.0/k)*inner(df("-")-df0("-"),phi("-"))*dS(5) - inner(u("-"), phi("-"))*dS(5)

    F_fluid += inner(J_(df)*sigma_f_new(u,p,df)*inv(F_(df)).T, grad(phi))*dx_f
    F_fluid -= inner(div(J_(df)*inv(F_(df)).T*u), gamma)*dx_f
    F_fluid += inner(J_(df("-"))*sigma_f_new(u("-"),p("-"),df("-"))*inv(F_(df("-"))).T*n("-"), phi("-"))*dS(5)
    F_fluid += inner(sigma_dev(d_("-"))*n("-"), phi("-"))*dS(5)
    F_fluid -= beta*h*h*inner(J_(df)*inv(F_(df).T)*grad(p), grad(gamma))*dx_f

    #F_fluid -= beta*h*h*inner(J_(df)*grad(p)*inv(F_(df)), grad(gamma))*dx_f

    af = lhs(F_fluid)
    bf = rhs(F_fluid)
    a_s = lhs(F_structure) #+ Mass_s_L#+ Mass_s_and_lhs
    b_s = rhs(F_structure) #+ Mass_s_and_rhs

    #print "b_s", type(b_s)


    adf = lhs(F_Ext)
    Ldf = rhs(F_Ext)
    return af, bf, a_s, b_s, adf, Ldf, M_lumped,Mass_s_and_rhs, M_time_lumped_lhs, Mass_s_rhs_L
