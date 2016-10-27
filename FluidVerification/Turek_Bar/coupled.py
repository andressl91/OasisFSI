from dolfin import *
import sys
import numpy as np
import matplotlib.pyplot as plt

#default values
v_deg = 1; p_deg = 1
solver = "Newton"; fig = False;

#command line arguments
while len(sys.argv) > 1:
    option = sys.argv[1]; del sys.argv[1];
    if option == "-v_deg":
        v_deg = int(sys.argv[1]); del sys.argv[1]
    elif option == "-p_deg":
        p_deg = int(sys.argv[1]); del sys.argv[1]
    elif option == "-solver":
        solver = str(sys.argv[1]); del sys.argv[1]
    elif option == "-fig":
        fig = bool(sys.argv[1]); del sys.argv[1]
    else:
        print sys.argv[0], ': invalid option', option

H = 0.41
D = 0.1
R = D/2.
Um = 0.2
nu = 0.001
rho = 10**3
mu = rho*nu


def fluid(mesh, solver, fig, v_deg, p_deg):
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

    bcs = [u_inlet, nos_geo, nos_wall]
    bcs2 = [u_inlet2, nos_geo, nos_wall]


    # TEST TRIAL FUNCTIONS
    phi, eta = TestFunctions(VQ)
    u ,p = TrialFunctions(VQ)

    ug, pg = TrialFunctions(VQ)
    phig, etag = TestFunctions(VQ)

    u0 = Function(V)


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

    if solver == "Newton":
        up = Function(VQ)
        u, p = split(up)

        up0 = Function(VQ)
        u0, p0 = split(up0)

        theta = 1.0

        F = (rho*theta*inner(dot(grad(u), u), phi) + rho*(1 - theta)*inner(dot(grad(u0), u0), phi)   \
        + inner(theta*sigma_f(p, u) + (1 - theta)*sigma_f(p0, u0), grad(phi) ) )*dx   \
        - eta*div(u)*dx

        if MPI.rank(mpi_comm_world()) == 0:
            print "Starting Newton iterations"

        J = derivative(F, up)

        problem = NonlinearVariationalProblem(F, up, bcs, J)
        solver  = NonlinearVariationalSolver(problem)

        prm = solver.parameters
        prm['newton_solver']['absolute_tolerance'] = 1E-6
        prm['newton_solver']['relative_tolerance'] = 1E-6
        prm['newton_solver']['maximum_iterations'] = 10
        prm['newton_solver']['relaxation_parameter'] = 1.0


        solver.solve()
        u_ , p_ = up.split(True)

        #plot(u_, interactive())

        file_v = File("velocity.pvd")
        file_v << u_

        file_p = File("pressure.pvd")
        file_p << p_

        drag, lift = integrateFluidStress(p_, u_)

        U_m = 2./3.*Um

        print('U_Dof= %d, cells = %d, v_deg = %d, p_deg = %d, \
        Drag = %f, Lift = %f' \
        % (V.dim(), mesh.num_cells(), v_deg, p_deg, drag, lift))

    if solver == "Newton2":

        up = Function(VQ)
        u, p = split(up)

        up0 = Function(VQ)
        u0, p0 = split(up0)

        theta = 1.0

        F = (rho*theta*inner(dot(grad(u), u), phi) + rho*(1 - theta)*inner(dot(grad(u0), u0), phi)   \
        + inner(theta*sigma_f(p, u) + (1 - theta)*sigma_f(p0, u0), grad(phi) ) )*dx   \
        + eta*div(u)*dx

        if MPI.rank(mpi_comm_world()) == 0:
            print "Starting Newton iterations"

        dw = TrialFunction(VQ)
        print "JACOBI"
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

        file_v = File("velocity.pvd")
        file_v << u_

        file_p = File("pressure.pvd")
        file_p << p_

        drag, lift = integrateFluidStress(p_, u_)

        U_m = 2./3.*Um

        print('U_Dof= %d, cells = %d, v_deg = %d, p_deg = %d, \
        Drag = %f, Lift = %f' \
        % (V.dim(), mesh.num_cells(), v_deg, p_deg, drag, lift))

    if solver == "Piccard":

        up = Function(VQ)

        if MPI.rank(mpi_comm_world()) == 0:
            print "Starting Piccard iterations"
            eps = 10
            k_iter = 0
            max_iter = 20

        while eps > 1E-7 and k_iter < max_iter:

            #SGIMA WRITTEN OUT
            #F = mu*inner(grad(u), grad(phi))*dx + rho*inner(grad(u)*u0, phi)*dx \
            #- div(phi)*p*dx - eta*div(u)*dx

            F = rho*inner(grad(u)*u0, phi)*dx +\
            inner(sigma_f(p, u), grad(phi))*dx- \
            eta*div(u)*dx

            solve(lhs(F) == rhs(F), up, bcs)
            u_ , p_ = up.split(True)
            eps = errornorm(u_, u0, degree_rise=3)
            u0.assign(u_)

            k_iter += 1

        if MPI.rank(mpi_comm_world()) == 0:
            print "iterations: %d  error: %.3e" %(k_iter, eps)

            u_ , p_ = up.split(True)
            #u_ , p_ = split(up)

            file_v = File("velocity.pvd")
            file_v << u_

            file_p = File("pressure.pvd")
            file_p << p_

            drag, lift = integrateFluidStress(p_, u_)

            U_m = 2./3.*Um

            print('U_Dof= %d, cells = %d, v_deg = %d, p_deg = %d, \
            Drag = %f, Lift = %f' \
            % (V.dim(), mesh.num_cells(), v_deg, p_deg, drag, lift))

#set_log_active(False)
for m in ["turek1.xml"]: #or turek1.xml
    mesh = Mesh(m)
    print "SOLVING FOR MESH %s" % m
    for i in range(2):
        if i > 0:
            mesh = refine(mesh)
            Drag = []; Lift = []; time = []
            fluid(mesh, solver, fig, v_deg, p_deg)
