from dolfin import *
import numpy as np
def solver_structure(N):
    mesh = UnitSquareMesh(N,N)
    V = VectorFunctionSpace(mesh,"CG",1)

    u = Function(V)
    psi = TestFunction(V)
    u1 = Function(V)
    bcs = []

    rho_s = 1.0E3
    mu_s = 0.5E6
    nu_s = 0.4
    lamda = nu_s*2*mu_s/(1-2*nu_s)

    I = Expression(("x[0]*x[0]+t*t","x[1]*x[1]+t*t"),t=0)
    u0 = Function(V)
    #f = Expression(("(2*x[0]+t*t)*(mu+0.5)+mu*(x[1]+t*t)+2","mu*(x[0]+t*t)+(mu+0.5*lam)*(2*x[1]+2*t)+2 "),mu=mu_s,lam=lamda,rho_s=rho_s,t=0)
    #f = Expression(("rho_s*2","rho_s*2"),rho_s=rho_s)
    f = Expression(("-24*lamda*x[0]*x[0] - 24*lamda*x[0] - 8*lamda*x[1]*x[1] - 8*lamda*x[1] - 4*lamda - 16*mu_s*x[0] - 8*mu_s + 2","-8*lamda*x[0]*x[0] - 8*lamda*x[0] - 24*lamda*x[1]*x[1] - 24*lamda*x[1] - 4*lamda - 16*mu_s*x[1] - 8*mu_s + 2"),mu_s =mu_s, lamda=lamda)

    dt = 0.001
    k = Constant(dt)

    def s_s_n_l(U):
        I = Identity(2)
        F = I + grad(U)
        E = 0.5*((F.T*F)-I)
        return lamda*tr(E)*I + 2*mu_s*E

    G =rho_s*((1./k**2)*inner(u - 2*u0 + u1,psi))*dx + inner(s_s_n_l(u0),grad(psi))*dx - inner(f,psi)*dx

    T = 1.0
    t=dt
    counter = 0
    e_list = np.zeros(10000)
    while t<=T:
        I.t = t
        u_ = interpolate(I,V)
        f.t = t
        print "Timestep: ",t
        solve(G==0,u,bcs,solver_parameters={"newton_solver": \
        {"relative_tolerance": 1E-6,"absolute_tolerance":1E-6,"maximum_iterations":100,"relaxation_parameter":1.0}})
        u0.assign(u1)
        u1.assign(u)
        e_list[counter] = errornorm(u, u_, norm_type="l2",degree_rise=3)
        print e_list[counter]
        t += dt
        counter += 1
    return e_list, mesh.hmin()


print solver_structure(100)


"""
N_list = [8,16,32,64]
E = np.zeros(len(N_list))
h = np.zeros(len(N_list))
for j in range(4):
    E[j],h[j]=solver_structure(N_list[j])
print E, h
for i in range(1, len(E)):
    r = ln(E[i]/E[i-1])/ln(h[i]/h[i-1])
    print "h=%10.2E r=%.2f" % (h[i], r)"""

"""
for i in range(len(N)):
    u ,u_exact,V,mesh = solver_1(N[i],mu,beta=1.0)
    #ex_norm = assemble((u_exact.dx(0) - u.dx(0))**2*dx)
    #ey_norm = assemble((u_exact.dx(1) - u.dx(1))**2*dx)

    #e = sqrt(mesh.hmin()*ex_norm + mu*(ex_norm+ey_norm))
    e = errornorm(u_exact, u, norm_type = "l2",degree_rise = 3)
    print e
    #u_norm = norm(u_exact,"H1")
    #e = errornorm(u_exact,u,norm_type = "l2",degree_rise = 3)
    b[i] = np.log(e)#/u_norm)
    a[i] = np.log(mesh.hmin())#1./(N[i]))
    #print u_norm
    #u_norm = sum(u_ex*u_ex)**0.5
A = np.vstack([a,np.ones(len(a))]).T
alpha_1,c_1 = np.linalg.lstsq(A,b)[0]

print "mu = ",mu, "C =  ", exp(c_1), "alpha =  ", alpha_1"""
