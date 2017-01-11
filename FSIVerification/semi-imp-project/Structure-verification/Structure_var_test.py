from dolfin import *
import numpy as np
from structur_sympy import *
import matplotlib.pyplot as plt
import sys
set_log_active(False)


def solver_structure(N,dt):
    if len(sys.argv) != 2:
        print "Usage: %s [implementation (1/2/3)]" % sys.argv[0]
        sys.exit(0)
    implementation = sys.argv[1]

    mesh = UnitSquareMesh(N,N)
    mesh2 = UnitSquareMesh(30,30)
    V = VectorFunctionSpace(mesh,"CG",1)
    V2 = VectorFunctionSpace(mesh2,"CG",2)


    class On(SubDomain):
    	def inside(self, x, on_boundary):
    		return on_boundary
    on = On()
    boundaries = FacetFunction("size_t",mesh)
    boundaries.set_all(0)
    on.mark(boundaries,1)
    I_1,f = find_my_f()

    #u_0 = Expression(("x[0]*x[0]","x[1]*x[1]"))

    #plot(boundaries,interactive=True)

    mu_s = 0.5E6
    nu_s = 0.4
    rho_s = 1.0E3
    lamda = nu_s*2*mu_s/(1-2*nu_s)

    k = Constant(dt)

    def s_s_n_l(U):
        I = Identity(2)
        F = I + grad(U)
        E = 0.5*((F.T*F)-I)
        return lamda*tr(E)*I + 2*mu_s*E

    if implementation =="1":
        bc1 = DirichletBC(V,I_1, boundaries,1)
        bcs = [bc1]
        psi = TestFunction(V)
        d = Function(V)
        d0 = Function(V)
        d1 = Function(V)
        d1=interpolate(I_1,V)
        I_1.t = dt
        d0 = interpolate(I_1,V)
        f.t = dt
        G =rho_s*((1./k**2)*inner(d - 2*d0 + d1,psi))*dx \
        + inner(s_s_n_l(0.5*(d+d1)),grad(psi))*dx - inner(f,psi)*dx
    elif implementation == "3":
        bc1 = DirichletBC(V, ((0,0)),boundaries, 1)
        bcs = [bc1]
        psi = TestFunction(V)
        w = Function(V)
        w0 = Function(V)
        d0 = Function(V)
        d = d0 + w*k

        G =rho_s*((1./k)*inner(w-w0,psi))*dx + rho_s*inner(dot(grad(0.5*(w+w0)),0.5*(w+w0)),psi)*dx \
        + inner(s_s_n_l(0.5*(d+d0)),grad(psi))*dx - inner(g,psi)*dx


    T = 1.0
    t=2*dt
    counter = 0
    e_list = []
    while t<=T:
        I_1.t = t
        d_ = interpolate(I_1,V2)
        f.t = t
        #f0.t = t-dt
        solve(G==0,d,bcs,solver_parameters={"newton_solver": \
        {"relative_tolerance": 1E-7,"absolute_tolerance":1E-7,"maximum_iterations":100,"relaxation_parameter":1.0}})
        #plot(u,mode="displacement")

        d_new = interpolate(d,V2)
        e_list.append(errornorm(d_, d_new, norm_type="l2",degree_rise=3))

        d1.assign(d0)
        d0.assign(d)
        print "Timestep: ",t#,"Error: ",e_list[counter]
        t += dt
        #counter += 1
    return np.mean(e_list), dt



#N_list = [2,4,8,16,32]

dt_list = [0.2,0.1,0.05,0.25,0.125, 0.0125/2.0]
E = np.zeros(len(dt_list))
h = np.zeros(len(dt_list))
for j in range(len(dt_list)):
    E[j],h[j]=solver_structure(N=20,dt=dt_list[j])
    #print "Error: %2.2E ,dt: %.f: "%(E[j],dt_list[j])
for i in range(1, len(E)):
    r = np.log(E[i]/E[i-1])/np.log(h[i]/h[i-1])
    print "h= %10.2E , r= %.6f" % (h[i], r)
for k in range(len(E)):
    print "Error: ", E[k]
