from dolfin import *
from mapping import *

def monolithic_form(VVQ,V1,V2,Q,dx_f,dx_s,mesh,v_deg,beta,n,lamda_s,mu_s,rho_f ,mu_f ,rho_s,dt):

    phi, psi, gamma = TestFunctions(VVQ)

    udp = Function(VVQ)
    u, d, p  = split(udp)

    udp0 = Function(VVQ)
    udp_res = Function(VVQ)
    d0 = Function(V2)
    d1 = Function(V2)
    u0 = Function(V1)
    p0 = Function(Q)

    k = Constant(dt)

    I = Identity(2)

    delta = 1.0E10
    h =  mesh.hmin()

    # Fluid variational form
    """
    F_fluid = (rho_f/k)*inner(J_(d)*(u - u0), phi)*dx_f
    F_fluid += rho_f*inner(J_(d)*grad(u)*inv(F_(d))*(u - ((d-d0)/k)), phi)*dx_f
    #F_fluid += rho_f*inner(J_(d)*inv(F_(d))*dot((u - ((d-d0)/k)),grad(u)), phi)*dx_f
    F_fluid -= inner(div(J_(d)*inv(F_(d))*u), gamma)*dx_f
    F_fluid += inner(J_(d)*sigma_f_new(u,p,d,mu_f)*inv(F_(d)).T, grad(phi))*dx_f
    """
    F_fluid = (rho_f/k)*inner(J_(d)*(u - u0), phi)*dx
    F_fluid += rho_f*inner(J_(d)*grad(u)*inv(F_(d))*(u - ((d-d0)/k)), phi)*dx
    F_fluid -= inner(div(J_(d)*inv(F_(d))*u), gamma)*dx
    F_fluid += inner(J_(d)*sigma_f_new(u,p,d,mu_f)*inv(F_(d)).T, grad(phi))*dx

    if v_deg == 1:
        F_fluid -= beta*h*h*inner(J_(d)*inv(F_(d).T)*grad(p), grad(gamma))*dx_f

        print "v_deg",v_deg

    # Structure var form
    F_structure = (rho_s/k)*inner(u-u0,phi)*dx_s + inner(P1(0.5*(d+d0),lamda_s,mu_s),grad(phi))*dx_s

    # Setting w = u on the structure using (d-d0)/k = w
    F_w = delta*((1.0/k)*inner(d-d0,psi)*dx_s - inner(0.5*(u+u0),psi)*dx_s)

    # laplace
    F_laplace =  inner(grad(d), grad(psi))*dx_f + (1./k)*inner(d-d0,psi)*dx_f  #- inner(grad(d)*n, psi)*ds

    F = F_fluid + F_structure + F_w + F_laplace
    return F, udp, udp_res, d0, d1, u0, p0
