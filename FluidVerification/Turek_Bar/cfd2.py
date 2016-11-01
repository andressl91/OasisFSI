from dolfin import *
import sys
import numpy as np
import matplotlib.pyplot as plt

import argparse
from argparse import RawTextHelpFormatter

parser = argparse.ArgumentParser(description="Implementation of Turek test case CFD1\n"
"For details: http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.550.1689&rep=rep1&type=pdf",\
 formatter_class=RawTextHelpFormatter, \
  epilog="############################################################################\n"
  "Example --> python cfd1.py -solver Newton2\n"
  "Example --> python cfd1.py -solver Newton -v_deg 2 -p_deg 1 -r  (Refines mesh one time, -rr for two etc.) \n"
  "############################################################################")
group = parser.add_argument_group('Parameters')
group.add_argument("-p_deg",  type=int, help="Set degree of pressure                     --> Default=1", default=1)
group.add_argument("-v_deg",  type=int, help="Set degree of velocity                     --> Default=2", default=2)
group.add_argument("-theta",  type=float, help="Explicit, Implicit, Cranc-Nic (0, 1, 0.5)  --> Default=1", default=2)
group.add_argument("-discr",  help="Write out or keep tensor in variational form --> Default=1", default="keep")
group.add_argument("-r", "--refiner", action="count", help="Mesh-refiner using built-in FEniCS method refine(Mesh)")
group2 = parser.add_argument_group('Solvers')
group2.add_argument("-solver", help="Newton   -- Fenics built-in module (DEFAULT SOLVER) \n"
"Newton2  -- Manuell implementation\n"
"Piccard  -- Manuell implementation\n", default="Newton")

args = parser.parse_args()

v_deg = args.v_deg
p_deg = args.p_deg
solver = args.solver
theta = args.theta
discr = args.discr
fig = False

#CFD1 Parameters
H = 0.41
D = 0.1
R = D/2.
Um = 1.0
nu = 0.001
rho = 10**3
mu = rho*nu


def fluid(mesh, solver, fig, v_deg, p_deg, theta):
    #plot(mesh)
    #interactive()

    V = VectorFunctionSpace(mesh, "CG", v_deg) # Fluid velocity
    Q  = FunctionSpace(mesh, "CG", p_deg)       # Fluid Pressure

    U_dof = V.dim()
    mesh_cells = mesh.num_cells()

    VQ = V*Q

    # BOUNDARIES

    Inlet  = AutoSubDomain(lambda x: "on_boundary" and near(x[0], 0))
    Outlet = AutoSubDomain(lambda x: "on_boundary" and near(x[0], 2.5))
    Walls  = AutoSubDomain(lambda x: "on_boundary" and near(x[1],0) or near(x[1], 0.41))

    boundaries = FacetFunction("size_t",mesh)
    boundaries.set_all(0)
    DomainBoundary().mark(boundaries, 1)
    Inlet.mark(boundaries, 2)
    Outlet.mark(boundaries, 3)
    Walls.mark(boundaries, 4)


    ds = Measure("ds", subdomain_data = boundaries)
    n = FacetNormal(mesh)
    #plot(boundaries,interactive=True)

    #BOUNDARY CONDITIONS


    ##UNSTEADY FLOW
    inlet = Expression(("1.5*Um*x[1]*(H - x[1]) / pow((H/2.0), 2)"\
    ,"0"), Um = Um, H = H)

    u_inlet = DirichletBC(VQ.sub(0), inlet, boundaries, 2)
    u_inlet2 = DirichletBC(VQ.sub(0), ((0, 0)), boundaries, 2)

    nos_geo = DirichletBC(VQ.sub(0), ((0, 0)), boundaries, 1)
    nos_wall = DirichletBC(VQ.sub(0), ((0, 0)), boundaries, 4)

    p_out = DirichletBC(VQ.sub(1), 0, boundaries, 3)

    bcs = [u_inlet, nos_geo, nos_wall, p_out]
    bcs2 = [u_inlet2, nos_geo, nos_wall]


    #Physical parameter
    t = 0.0

    #MEK4300 WAY
    def FluidStress(p, u):
        print "MEK4300 WAY"
        n = -FacetNormal(mesh)
        n1 = as_vector((1.0,0)) ; n2 = as_vector((0,1.0))
        nx = dot(n,n1) ; ny = dot(n,n2)
        nt = as_vector((ny,-nx))

        ut = dot(nt, u)
        Fd = assemble((rho*nu*dot(grad(ut),n)*ny - p*nx)*ds(1))
        Fl = assemble(-(rho*nu*dot(grad(ut),n)*nx + p*ny)*ds(1))

        return Fd, Fl


    #MY WAY
    def integrateFluidStress(p, u):
        print "MY WAY!"

        eps   = 0.5*(grad(u) + grad(u).T)
        sig   = -p*Identity(2) + 2.0*mu*eps

        traction  = dot(sig, -n)

        forceX  = traction[0]*ds(1)
        forceY  = traction[1]*ds(1)
        fX      = assemble(forceX)
        fY      = assemble(forceY)

        return fX, fY

    def sigma_f(p, u):
        return - p*Identity(2) + mu*(grad(u) + grad(u).T)

    Re = Um*D/nu
    print "SOLVING FOR Re = %f" % Re #0.1 Cylinder diameter
    print "Method %s" % (solver)

    if solver == "Newton" or solver == "Newton2":
        phi, eta = TestFunctions(VQ)
        up = Function(VQ)
    	u, p = split(up)
        #For non-theta
        #u0 = Function(V)

        up0 = Function(VQ)
    	u0, p0 = split(up0)

        if discr == "keep":
            F = (rho*(theta*inner(dot(grad(u), u), phi) + (1 - theta)*inner(dot(grad(u0), u0), phi) ) \
                + inner(theta*sigma_f(p, u) + (1 - theta)*sigma_f(p0, u0), grad(phi)) ) *dx \
                - eta*div(u)*dx

        if discr == "split":
    		F =   rho*inner(theta*grad(u)*u + (1 -theta)*grad(u0)*u0, phi) *dx + \
    			  mu*inner(theta*grad(u) + (1-theta)*grad(u0) , grad(phi))*dx - \
    			  (theta*div(phi)*p + (1 - theta)*div(phi)*p0)*dx - eta*div(u)*dx


    if solver == "Newton":
        if MPI.rank(mpi_comm_world()) == 0:
            print "Starting Newton iterations"

        J = derivative(F, up)

        problem = NonlinearVariationalProblem(F, up, bcs, J)
        solver  = NonlinearVariationalSolver(problem)

        prm = solver.parameters
        prm['nonlinear_solver'] = 'newton'
        #info(prm,True)  #get full info on the parameters
        #list_linear_solver_methods()#Linear solvers
        prm['newton_solver']['absolute_tolerance'] = 1E-6
        prm['newton_solver']['relative_tolerance'] = 1E-6
        prm['newton_solver']['maximum_iterations'] = 10
        prm['newton_solver']['relaxation_parameter'] = 1.0
        prm['newton_solver']['linear_solver'] = 'mumps'

        tic()
        solver.solve()
        print "Solving time %g" % toc()
        u_ , p_ = up.split(True)

        drag, lift = integrateFluidStress(p_, u_)

        U_m = 2./3.*Um

        print('U_Dof= %d, cells = %d, v_deg = %d, p_deg = %d, \
        Drag = %f, Lift = %f, discretisation = %s' \
        % (V.dim(), mesh.num_cells(), v_deg, p_deg, drag, lift, discr))

    if solver == "Newton2":

        if MPI.rank(mpi_comm_world()) == 0:
            print "Starting Manual implemented Newton iterations"

        dw = TrialFunction(VQ)
        dF_W = derivative(F, up)                # Jacobi

        atol, rtol = 1e-7, 1e-6                  # abs/rel tolerances
        lmbda      = 1.0                            # relaxation parameter
        WD_inc      = Function(VQ)                  # residual
        Iter      = 0                               # number of iterations
        residual   = 1                              # residual (To initiate)
        rel_res    = residual                       # relative residual
        max_it    = 100                              # max iterations
        bcs_u = []

        for bc in bcs:
            bc.apply(up.vector())

        for i in bcs:
            i.homogenize()
            bcs_u.append(i)

        while rel_res > rtol and residual > atol and Iter < max_it:
            A, b = assemble_system(dF_W, -F, bcs_u)

            # Must be implemented in FSI #############
            #A.ident_zeros()                         #
            #[bc.apply(A, b) for bc in bcs_u]        #
            ##########################################
            solve(A, WD_inc.vector(), b)

            rel_res = norm(WD_inc, 'l2')

            a = assemble(F)
            for bc in bcs_u:
                bc.apply(a)
            residual = b.norm('l2')

            up.vector()[:] += lmbda*WD_inc.vector()
            print "Newton iteration %d: r (atol) = %.3e (tol = %.3e), r (rel) = %.3e (tol = %.3e) " \
            % (Iter, residual, atol, rel_res, rtol)
            Iter += 1

        #print type(d)
        for bc in bcs:
            bc.apply(up.vector())

        u_, p_  = up.split(True)

        drag, lift = integrateFluidStress(p_, u_)

        U_m = 2./3.*Um

        print('U_Dof= %d, cells = %d, v_deg = %d, p_deg = %d, \
        Drag = %f, Lift = %f, discretisation = %s' \
        % (V.dim(), mesh.num_cells(), v_deg, p_deg, drag, lift, discr))

    if solver == "Piccard":

        phi, eta = TestFunctions(VQ)
        u ,p = TrialFunctions(VQ)

        u0 = Function(V); p0 = Function(Q)
        up = Function(VQ)

        if MPI.rank(mpi_comm_world()) == 0:
            print "Starting Piccard iterations"

            tol    = 1E-6                            # tolerance
            Iter   = 0                               # number of iterations
            eps    = 1                               # residual (To initiate)
            max_it = 100                             # max iterations

        while eps > tol and Iter < max_it:

            F = (rho*theta*inner(dot(grad(u), u0), phi) + rho*(1 - theta)*inner(dot(grad(u0), u0), phi)   \
            + inner(theta*sigma_f(p, u) + (1 - theta)*sigma_f(p0, u0), grad(phi) ) )*dx   \
            + eta*div(u)*dx

            solve(lhs(F) == rhs(F), up, bcs)
            u_ , p_ = up.split(True)
            eps = errornorm(u_, u0, norm_type="l2", degree_rise=2)
            #test = u_ - u0
            #print norm(test, 'l2')
            u0.assign(u_)
            print "iterations: %d  error: %.3e" %(Iter, eps)

            Iter += 1

        if MPI.rank(mpi_comm_world()) == 0:
            u_ , p_ = up.split(True)

            drag, lift = integrateFluidStress(p_, u_)

            U_m = 2./3.*Um

            print('U_Dof= %d, cells = %d, v_deg = %d, p_deg = %d, \
            Drag = %f, Lift = %f, discretisation = %s' \
            % (V.dim(), mesh.num_cells(), v_deg, p_deg, drag, lift, discr))

#set_log_active(False)

mesh = Mesh("turek1.xml")
Drag = []; Lift = []; time = []
if args.refiner == None:
    fluid(mesh, solver, fig, v_deg, p_deg, theta)

else:
    for i in range(args.refiner):
        mesh = refine(mesh)
    fluid(mesh, solver, fig, v_deg, p_deg, theta)
