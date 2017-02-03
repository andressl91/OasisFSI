from dolfin import *
import matplotlib.pyplot as plt
import numpy as np
# Mesh
import argparse
from Find_f import *
v_deg = 2
p_deg = 1
w_deg = 2
T = 1.0
test = 1
def ALE_fluid_solver(N,dt):
    mesh = RectangleMesh(Point(0,0), Point(2, 1), N, N, "crossed")
    mesh2 = RectangleMesh(Point(0,0), Point(2, 1), 40, 40, "crossed")
    # FunctionSpaces
    V1 = VectorFunctionSpace(mesh, "CG", v_deg) # Fluid velocity
    V2 = VectorFunctionSpace(mesh, "CG", w_deg) # fluid displacement velocity
    Q = FunctionSpace(mesh, "CG", p_deg)
    VVQ = MixedFunctionSpace([V1, Q])
    V3 = VectorFunctionSpace(mesh2, "CG",2)

    # Boundaries

    class On(SubDomain):
    	def inside(self, x, on_boundary):
    		return on_boundary
    on = On()
    boundaries = FacetFunction("size_t",mesh)
    boundaries.set_all(0)
    on.mark(boundaries,1)
    #plot(boundaries,interactive=True)
    ds = Measure("ds", subdomain_data = boundaries)
    n = FacetNormal(mesh)

    I_u ,f = find_my_f()
    I_w = Expression(("cos(t)*(x[0]-2) ","0"),t=0)

    bc1 = DirichletBC(V1, I_u, boundaries,1)
    bc2 = DirichletBC(V2, I_w, boundaries,1)

    bcs_w = [bc2]
    bcs_u = [bc1]

    # Test and Trial functions
    phi, gamma = TestFunctions(VVQ)
    psi = TestFunction(V2)
    w = TrialFunction(V2)
    up_ = Function(VVQ)
    u, p = split(up_)
    u0 = Function(V1)
    d = Function(V2)
    w_ = Function(V2)
    k = Constant(dt)

    # Fluid properties
    rho_f   = Constant(1.0E3)
    nu_f = Constant(1.0E-3)
    mu_f    = Constant(1.0)
    I = Identity(2)
    def Eij(U):
    	return sym(grad(U))# - 0.5*dot(grad(U),grad(U))

    def F_(U):
    	return (I + grad(U))

    def J_(U):
    	return det(F_(U))

    def E(U):
    	return 0.5*(F_(U).T*F_(U)-I)

    def S(U):
    	return (2*mu_s*E(U) + lamda_s*tr(E(U))*I)

    def P1(U):
    	return F_(U)*S(U)

    def sigma_f(v,p):
    	return 2*mu_f*sym(grad(v)) - p*Identity(2)

    def sigma_s(u):
    	return 2*mu_s*sym(grad(u)) + lamda_s*tr(sym(grad(u)))*I

    def sigma_f_hat(v,p,u):
    	return J_(u)*sigma_f(v,p)*inv(F_(u)).T
    def epsilon(u):
        return 0.5*(grad(u) + grad(u).T)
    def sigma_f_new(u,p,d):
    	return -p*I + mu_f*(grad(u)*inv(F_(d)) + inv(F_(d)).T*grad(u).T)

    d = k*w_

    # Fluid variational form
    if test == 1: # The richter way
        u0 = interpolate(I_u,V1)
        f.t = dt
        F_fluid = (rho_f/k)*inner(J_(d)*(u - u0), phi)*dx
        F_fluid += rho_f*inner(J_(d)*inv(F_(d))*dot((u - w_),grad(u)), phi)*dx
        F_fluid -= inner(div(J_(d)*inv(F_(d))*u), gamma)*dx
        F_fluid += inner(J_(d)*sigma_f_new(u,p,d)*inv(F_(d)).T, grad(phi))*dx
        F_fluid -= inner(J_(d)*f,phi)*dx
    if test == 3: # The richter way with tensor written out
        F_fluid = (rho_f/k)*inner(J_(d)*(u - u0), phi)*dx
        F_fluid += rho_f*inner(J_(d)*inv(F_(d))*(u - w_)*grad(u), phi)*dx
        F_fluid -= inner(div(J_(d)*inv(F_(d))*u), gamma)*dx
        F_fluid += inner(J_(d)*2*mu_f*epsilon(u)*inv(F_(d))*inv(F_(d)).T ,epsilon(phi) )*dx
        F_fluid -= inner(J_(d)*p*I*inv(F_(d)).T, grad(phi))*dx

    if test == 3: # Written
        F_fluid = (rho_f/k)*inner(J_(d)*(u - u0), phi)*dx
        F_fluid += rho_f*inner(J_(d)*grad(u)*inv(F_(d))*(u - w_), phi)*dx
        F_fluid -= inner(div(J_(d)*inv(F_(d))*u), gamma)*dx
        F_fluid += inner(J_(d)*2*mu_f*epsilon(u)*inv(F_(d))*inv(F_(d)).T ,epsilon(phi) )*dx
        F_fluid -= inner(J_(d)*p*I*inv(F_(d)).T, grad(phi))*dx

    if test == 4: # Written with tensor and grad on the left side.
        F_fluid = (rho_f/k)*inner(J_(d)*(u - u0), phi)*dx
        F_fluid += rho_f*inner(J_(d)*grad(u)*inv(F_(d))*(u - w_), phi)*dx
        F_fluid -= inner(div(J_(d)*inv(F_(d))*u), gamma)*dx
        F_fluid += inner(J_(d)*sigma_f_new(u,p,d)*inv(F_(d)).T, grad(phi))*dx

    # laplace d = 0
    F2 =  k*(inner(grad(w), grad(psi))*dx - inner(grad(w)*n, psi)*ds)

    u_file = XDMFFile(mpi_comm_world(), "new_results/velocity.xdmf")
    d_file = XDMFFile(mpi_comm_world(), "new_results/d.xdmf")
    w_file = XDMFFile(mpi_comm_world(), "new_results/w.xdmf")
    p_file = XDMFFile(mpi_comm_world(), "new_results/pressure.xdmf")

    for tmp_t in [u_file, d_file, p_file, w_file]:
        tmp_t.parameters["flush_output"] = True
        tmp_t.parameters["multi_file"] = 1
        tmp_t.parameters["rewrite_function_mesh"] = False

    time_array = []
    flux = []
    e_list = []
    t = dt
    while t <= T:
        time_array.append(t)
        solve(lhs(F2)==rhs(F2), w_, bcs_w)
        I_u.t = t
        I_w.t = t
        u_exact = interpolate(I_u,V3)
        w_exact = interpolate(I_w,V3)
        f.t = t
        solve(F_fluid==0, up_, bcs_u,solver_parameters={"newton_solver": \
        {"relative_tolerance": 1E-8,"absolute_tolerance":1E-8,\
        "maximum_iterations":100,"relaxation_parameter":1.0}})
        u,p = up_.split(True)

        u_new = interpolate(u,V3)
        error = errornorm(u_exact, u_new, norm_type="l2",degree_rise=3)
        e_list.append(error)
        print "Time: ",t
        print "Error: ",error

        u0.assign(u)
        u_file << u
        #p_file << p
        #w_file << w_

        t += dt
    return np.mean(e_list), dt
#print ALE_fluid_solver(N=20,dt=0.001)
#N_list = [16,32,64]

dt_list = [0.2E-3,0.1E-3,0.05E-3,0.025E-3]
E = np.zeros(len(dt_list))
h = np.zeros(len(dt_list))
for j in range(len(dt_list)):
    E[j],h[j] = ALE_fluid_solver(N=5,dt=dt_list[j])
    #print "Error: %2.2E ,dt: %.f: "%(E[j],dt_list[j])
for i in range(1, len(E)):
    r = np.log(E[i]/E[i-1])/np.log(h[i]/h[i-1])
    print "h= %10.2E , r= %.6f" % (h[i], r)
for k in range(len(E)):
    print "Error: ", E[k]
#print len(flux),len(time_array)
#plt.plot(time_array,flux);plt.title("Flux, with N = 20"); plt.ylabel("Flux out");plt.xlabel("Time");plt.grid();
#plt.show()
