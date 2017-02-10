#from ufl import *
from fenics import *
#from dolfin import *
import numpy as np
import matplotlib.pyplot as plt
set_log_active(True)

def ALE_run(N,dt):
	rho_f = 1.0
	mu_f = 1.0
	I = Identity(2)
	mesh = UnitSquareMesh(N,N)
	x = SpatialCoordinate(mesh)
	V = VectorFunctionSpace(mesh, "CG", 2) #fluid velocity function space
	Q = FunctionSpace(mesh, "CG", 1) # fluid pressure space
	VVQ = MixedFunctionSpace([V, Q])

	phi, gamma = TestFunctions(VVQ)

	up = Function(VVQ)
	u, p  = split(up)

	up0 = Function(VVQ)
	up_res = Function(VVQ)
	d = Function(V)
	w = Function(V)
	d0 = Function(V)
	d1 = Function(V)
	u0 = Function(V)
	p0 = Function(Q)

	def sigma_f(v,p):
		return mu_f*(grad(u) + grad(u).T) - p*Identity(2)
	def F_(U):
		return (I + grad(U))

	def J_(U):
		return det(F_(U))
	def sigma_f_new(u,p,d):
		return -p*I + mu_f*(grad(u)*inv(F_(d)) + inv(F_(d)).T*grad(u).T)

	def Newton_manual(F, up, bcs, atol, rtol, max_it, lmbda,up_res,VVQ):
	    #Reset counters
	    Iter      = 0
	    residual   = 1
	    rel_res    = residual
	    dw = TrialFunction(VVQ)
	    Jac = derivative(F, up,dw)                # Jacobi

	    while rel_res > rtol and residual > atol and Iter < max_it:
	        A = assemble(Jac)
	        A.ident_zeros()
	        b = assemble(-F)

	        [bc.apply(A, b, up.vector()) for bc in bcs]

	        #solve(A, udp_res.vector(), b, "superlu_dist")

	        solve(A, up_res.vector(), b)#, "mumps")

	        up.vector()[:] = up.vector()[:] + lmbda*up_res.vector()[:]
	        #udp.vector().axpy(1., udp_res.vector())
	        [bc.apply(up.vector()) for bc in bcs]
	        rel_res = norm(up_res, 'l2')
	        residual = b.norm('l2')

	        if MPI.rank(mpi_comm_world()) == 0:
	            print "Newton iteration %d: r (atol) = %.3e (tol = %.3e), r (rel) = %.3e (tol = %.3e) " \
	        % (Iter, residual, atol, rel_res, rtol)
	        Iter += 1

	    return up

	u_e = Expression(("x[1]*cos(x[1])","x[0]*cos(x[0])"),t=0)
	d_e = Expression(("(1./2.)*x[0]*x[0]","-(1./2.)*x[1]*x[1]"),t=0)
	w_e = Expression(("x[1]*cos(x[1])","x[0]*cos(x[0])"),t=0)
	p_e = Expression("2",t=0)

	u0 = interpolate(u_e,V)
	d = interpolate(d_e,V)
	w = interpolate(u_e,V)

	#f = Expression(("-x[0]*x[0]*sin(t)-2*cos(t)","x[1]*x[1]*cos(t)-2*sin(t)"), t = 0)
	k = Constant(dt)
	t = variable(Constant(dt))
	u_x = x[1]*cos(x[1])
	u_y = x[0]*cos(x[0])
	d_x = 1/2*x[0]*x[0]*t
	d_y = -1/2*x[1]*x[1]*t

	p_ = 2

	u_vec = as_vector([u_x, u_y])
	d_vec = as_vector([d_x, d_y])
	w_vec = as_vector([u_x, u_y])

	#Sourceterm for ALE:
	#F_fluid_source = rho_f*diff(u_vec, t)
	F_fluid_source = rho_f*grad(u_vec)*(u_vec - w_vec)
	#F_fluid_source -= div(grad(u_vec))
	F_fluid_source -= div(sigma_f(u_vec,p_))

	bc1 = DirichletBC(VVQ.sub(0), u_e, "on_boundary")
	bc2 = DirichletBC(VVQ.sub(1), p_e, "on_boundary")
	#bc2 = DirichletBC(W, u_e, "x[0] > 1 - DOLFIN_EPS")
	bcs = [bc1,bc2]

	#F_fluid = (rho_f/k)*inner(u - u0, phi)*dx
	F_fluid = rho_f*inner(grad(u)*(u - w), phi)*dx
	#F_fluid += rho_f*inner(dot((u - ((d-d0)/k)),grad(u)), phi)*dx
	F_fluid -= inner(div(u), gamma)*dx
	F_fluid += inner(sigma_f(u,p), grad(phi))*dx
	F_fluid -= inner(F_fluid_source, phi)*dx
	"""
	#F_fluid = (rho_f/k)*inner(J_(d)*(u - u0), phi)*dx
	F_fluid = rho_f*inner(J_(d)*grad(u)*inv(F_(d))*(u - w), phi)*dx
	#F_fluid = rho_f*inner(J_(d)*inv(F_(d))*dot((u - ((d-d0)/k)),grad(u)), phi)*dx
	F_fluid -= inner(div(J_(d)*inv(F_(d))*u), gamma)*dx
	F_fluid += inner(J_(d)*sigma_f_new(u,p,d)*inv(F_(d)).T, grad(phi))*dx
	F_fluid -= inner(J_(d)*F_fluid_source, phi)*dx"""

	T = 0.0001
	t_step = dt
	error = []
	atol = 1e-6;rtol = 1e-6; max_it = 100; lmbda = 1.0;
	#while t_step <= T:
		#print "Solving for time %f" % t_step
	#u_e.t = t_step
	#d_e.t = t_step
	#w_e.t = t_step
	#t = variable(Constant(t_step))

	#print "ERRORNORM", errornorm(u_e, u_sol, norm_type="l2", degree_rise = 3)

	print "Time t = %.5f" % t
	#Reset counters
	up = Newton_manual(F_fluid, up, bcs, atol, rtol, max_it, lmbda,up_res,VVQ)
	#solve(F_fluid==0, up, bcs)
	u,p = up.split(True)

	E.append(errornorm(u_e, u, norm_type="l2", degree_rise = 2))

	#plot(u)
	#u0.assign(u)
	#d1.assign(d0)
	#d0.assign(d)
	#p0.assign(p)
	#t_step += dt
	#E.append(error)
	h.append(N)
#N = [8]
#dt = [0.1, 0.09, 0.08, 0.07]
dt = [0.000001]
N = [2**i for i in range(0, 5)]
#N = [4]
E = []; h = []
for n in N:
    print "Solving for N : %d" % n
    for t in dt:
        print "Solving for dt: %g" % t
        ALE_run(n, t)

for i in E:
    print i

print "Convergence rate"
for i in range(len(E) - 1):
    r = np.log(E[i+1]/E[i] ) / np.log(h[i+1]/h[i])
    #r = np.log(E[i+1]/E[i] ) / np.log(dt[i+1]/dt[i])
    print r
