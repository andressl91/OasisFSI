from dolfin import *
import mshr
from cbcflow import *
from cbcflow.schemes.utils import *
from cbcpost.utils.mpi_utils import gather, broadcast
from cbcpost import *
from cbcpost.utils import *
from utils import *
import numpy as np
from IPython import embed
import sys

class Left(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[0], 0)

class Right(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[0], 5.0)

class Extrapolation(Function):      # {{{
    "Helper class for creating an extrapolated function"
    def __init__(self, V, k):
        Function.__init__(self, V)
        self.k = k
        self.funcs = [Function(V) for i in range(k)]

    def update(self, f):
        if self.k == 0:
            return
        for i in range(self.k-1):
            self.funcs[i] = self.funcs[i+1]

        self.funcs[-1].assign(f)
        self.vector().zero()

        if self.k == 1:
            self.vector().axpy(1.0, self.funcs[0].vector())
        elif self.k == 2:
            self.vector().axpy(2.0, self.funcs[0].vector())
            self.vector().axpy(-1.0, self.funcs[1].vector())
    # }}}


class FSI_Decoupled(NSScheme):
    @classmethod
    def default_params(cls):
        params = NSScheme.default_params()
        params.update(
            # Default to P2-P1
            u_degree = 2,
            p_degree = 1,
            r = 2, # Extrapolation degree of velocity
            s = 1, # Extrapolation degree of pressure and displacement
            )
        return params
    
    def solve(self, problem, timer):
        assert isinstance(problem, FSIProblem)

        mesh = problem.mesh
        bmesh = BoundaryMesh(mesh, "exterior")

        dim = mesh.geometry().dim()

        dx = problem.dx
        ds = problem.ds

        n = FacetNormal(mesh)

        dt, timesteps, start_timestep = compute_regular_timesteps(problem)
        dt = Constant(dt)
        t = Constant(timesteps[start_timestep], name="TIME")

        # Function spaces
        spaces = NSSpacePoolSplit(mesh, self.params.u_degree, self.params.p_degree)
        V = spaces.V
        Q = spaces.Q
        D = spaces.spacepool.get_custom_space("CG", 1, (dim,))

        Db = spaces.spacepool.get_custom_space("CG", 1, (dim,), boundary=True)
        Dgb = spaces.spacepool.get_custom_space("CG", 1, (dim,), boundary=True)

        # Trial- and testfunctions
        u, v = TrialFunction(V), TestFunction(V) # Velocity
        p, q = TrialFunction(Q), TestFunction(Q) # Pressure
        eta, x = TrialFunction(Dgb), TestFunction(Dgb) # Boundary displacement
        d, e = TrialFunction(D), TestFunction(D) # Mesh displacement

        # Solution functions
        U = Function(V) # Velocity
        P = Function(Q) # Pressure
        DF = Function(D) # Fluid (full mesh) displacement

        # Helper functions
        U1 = Function(V)
        U2 = Function(V)
        DF1 = Function(D)
        DF2 = Function(D)
        ETA1 = Function(Dgb)        # eta time derivatives
        
        Uext = Extrapolation(V, self.params.r) # Velocity extrapolation
        Pext = Extrapolation(Q, self.params.s) # Pressure extrapolation
        phi = p - Pext
        phiext = Extrapolation(Q, self.params.r)
        DFext = Extrapolation(D, self.params.s)
        DFext1, DFext2 = Function(D), Function(D)
        w = Function(D)
        db = Function(Db)
        dgb = Function(Dgb)
        dgb1, dgb2 = Function(Dgb), Function(Dgb)
        traction = Function(Dgb)

        # Get functions for data assimilation
        observations = problem.observations(spaces, t)
        controls = problem.controls(spaces)

        # Make scheme-specific representation of bcs
        bcs = problem.boundary_conditions(spaces, U, P, t, None)
        bcu = make_velocity_bcs(problem, spaces, bcs)
        bcp = make_pressure_bcs(problem, spaces, bcs)

        bcs_eta = [DirichletBC(D, DF, DomainBoundary())]    # Are these for eta or full (fluid) mesh displacement?
        bcs_eta += [DirichletBC(D, expr, problem.facet_domains, i) for expr, i in bcs[2]]

        # Dgb defined in bdry_mesh, facet domains on mesh
        # Create a facet
        dgb_mesh = Dgb.mesh()
        dgb_dim = dgb_mesh.topology().dim() 
        eta_boundaries = MeshFunction('size_t', dgb_mesh, dgb_dim - 1)
        eta_boundaries.set_all(0)
        Left().mark(eta_boundaries, 1)
        Right().mark(eta_boundaries, 2)
        #plot(eta_boundaries, interactive=True)
        #embed()
        #sys.exit()

        #bcs_shell = [DirichletBC(Dgb, expr, DomainBoundary(), method="pointwise") for expr, i in bcs[2]] # really for eta
        #shellbc1 = DirichletBC(Dgb, Constant((0, 0)), eta_boundaries, 1, method="pointwise")
        #shellbc2 = DirichletBC(Dgb, Constant((0, 0)), eta_boundaries, 2, method="pointwise")
        shellbc1 = DirichletBC(Dgb, Constant((0, 0)), eta_boundaries, 1)
        shellbc2 = DirichletBC(Dgb, Constant((0, 0)), eta_boundaries, 2)
        bcs_shell = [shellbc1, shellbc2]
        #embed()
        #sys.exit()

        # Create boundary integrals over structure and non-structure part of mesh
        # nds = non-fixed part of boundary
        # fds = fixed part of boundary (has bcs for displacement)
        fixed_boundaries = [i for _,i in bcs[2]]        # bc2 is bceta
        nonfixed_boundaries = list(set(np.unique(problem.facet_domains.array())) - set(fixed_boundaries))
        nonfixed_boundaries = gather(nonfixed_boundaries, 0)
        nonfixed_boundaries = np.unique(np.array(nonfixed_boundaries))
        nonfixed_boundaries = broadcast(nonfixed_boundaries, 0)
        nonfixed_boundaries = nonfixed_boundaries.astype(np.int)
        assert len(nonfixed_boundaries) > 0

        nds = ds(nonfixed_boundaries[0])        # Is nds Sigma?
        for i in nonfixed_boundaries[1:]: 
            nds += ds(i)
        if len(fixed_boundaries) > 0:
            fds = ds(fixed_boundaries[0])       # Is fds Gamma?
            for i in fixed_boundaries[1:]: 
                fds += ds(i)

        def par(f,n):
            "Return parallel/tangential component of f"
            return f-dot(f,n)*n

        mu = Constant(problem.params.mu) # Fluid viscosity
        rho_f = Constant(problem.params.rho) # Fluid density
        rho_s = Constant(problem.params.rho_s) # Structure density
        h_s = Constant(problem.params.h) # Thickness
        
        if isinstance(problem.params.R, float):
            R = Constant(problem.params.R)
        else:
            R = problem.params.R

        beta = Constant(problem.params.E*float(h_s)/(1 - problem.params.nu**2)*1.0/R**2)
        beta1 = Constant(0.5*problem.params.E*float(h_s)/(1 + problem.params.nu))
        alpha = Constant(1)
        alpha1 = Constant(1e-3)

        # Set up Field (see cbcpost documentation) to compute boundary traction
        T_bdry = BoundaryTraction()
        T_bdry.before_first_compute(lambda x: U)

        r = self.params.r
        s = self.params.s

        I = SpatialCoordinate(mesh)
        
        A = I+DF # ALE map
        F = grad(A)
        J = det(F)

        # Tentative velocity, Eq. 30, mapped to reference mesh
        # Eq. 30.1
        a1 = inner(rho_f*(u-U1)/dt, v)*J*dx() # Is this mapping correct?
        #a1 = inner(rho_f*(u-U1)/dt, v)*dx() # Is this mapping correct?

        a1 += rho_f*inner((grad(u)*inv(F))*(U1 - w), v)*J*dx() 
        #a1 += rho_f*inner(grad(u)*U1, v)*dx() 

        a1 += inner(J*Sigma(mu, u, Pext, F)*inv(F).T, grad(v))*dx()
        #a1 += inner(sigma(mu, u, Pext), grad(v))*dx()
        a1 -= inner(J*Sigma(mu, u, Pext, F)*inv(F).T*n, v)*ds()
        #a1 -= inner(sigma(mu, u, Pext)*n, v)*ds()

        a1 += rho_f*inner((grad(u)*inv(F))*(U1-w), v)*J*dx()    # dot or *

        a1 += inner(J*Sigma(mu,u,Pext, F)*inv(F).T,grad(v))*dx()

        # Eq. 30.2
        #a1 += inner(sigma(mu, u, Pext)*n, v)*fds # Inflow/outflow. Should be Sigma?
        a1 += inner(Sigma(mu, u, Pext, F)*n, v)*fds # Matters only if Gamma is non-fixed? 
        #a1 += inner(sigma(mu, u, Pext)*n, v)*fds # Matters only if Gamma is non-fixed? 

        # Eq. 30.3
        #a1 += inner(J*sigma(mu, u, Pext)*inv(F).T*n, v)*nds # Should be Sigma?
        a1 += inner(J*Sigma(mu, u, Pext, F)*inv(F).T*n, v)*nds # Sigma because boundary moves?
        a1 += inner(sigma(mu, u, Pext)*n, v)*nds # Sigma because boundary moves?
        a1 += rho_s*h_s/dt*inner(u, v)*nds # Should contain J?
        a1 -= rho_s*h_s/dt*inner((DF1 - DF2)/dt + par(dt*(DFext - 2*DFext1 + DFext2)/dt**2, n), v)*nds

        # Extrapolation cases of RHS of Eq. 30.3
        if r == 1:
            _F = grad(I + DF1)
            _J = det(_F)
            #a1 -= 2*mu*inner(par(_J*Epsilon(U1, _F)*inv(_F).T*n, n), v)*nds
            a1 -= 2*mu*inner(par(epsilon(U1)*n, n), v)*nds
        elif r == 2:
            _F = grad(I + DF1)
            _J = det(_F)
            a1 -= 2*2*mu*inner(par(_J*Epsilon(U1, _F)*inv(_F).T*n, n), v)*nds
            
            _F = grad(I + DF2)
            _J = det(_F)
            a1 += 2*mu*inner(par(_J*Epsilon(U2, _F)*inv(_F).T*n, n), v)*nds
        
        L1 = rhs(a1)
        a1 = lhs(a1)
        
        A1 = assemble(a1)
        b1 = assemble(L1)
        
        solver_u_tent = create_solver("gmres", "additive_schwarz")

        # Pressure, Eq 31, mapped to reference mesh (note phi=p - Pext)
        # Eq. 31.1

        #a2 = inner(J*inv(F)*inv(F).T*grad(phi), grad(q))*dx()
        a2 = inner(dt/rho_f*J*inv(F)*inv(F).T*grad(phi), grad(q))*dx()
        #a2 = inner(dt/rho_f*grad(phi), grad(q))*dx()
        #a2 -= inner(J*inv(F)*inv(F).T*grad(phi), q*n)*ds()
        a2 -= inner(dt/rho_f*J*inv(F)*inv(F).T*grad(phi), q*n)*ds()
        #a2 -= inner(dt/rho_f*grad(phi), q*n)*ds()
        a2 += div(J*inv(F)*U)*dx()
        #a2 += div(U)*dx()

        # Eq. 31.2
        a2 += phi*q*fds
        a2 -= (P - Pext)*q*fds

        # Eq. 31.3
        # Why no J?
        #a2 += dt/rho_f*dot(dot(grad(phi), n), q)*nds
        #a2 += dt/(rho_s*h_s)*phi*q*nds
        #a2 -= dt/(rho_s*h_s)*phiext*q*nds
        #a2 -= inner(Uext - (DFext - DFext1)/dt, n)*q*nds

        a2 += dt/rho_f*J*dot(dot(inv(F).T*grad(phi), inv(F).T*n), q)*nds
        a2 += dt/(rho_s*h_s)*J*phi*q*nds
        a2 -= dt/(rho_s*h_s)*J*phiext*q*nds
        a2 -= J*inner(Uext - (DFext - DFext1)/dt, inv(F).T*n)*q*nds

        # J*grad(phi)*inf(F), grad(q)*inf(F)
        a2 = inner(J*inv(F)*inv(F).T*grad(phi), grad(q))*dx()   # symmetry and *
        a2 -= inner(J*inv(F)*inv(F).T*grad(phi), q*n)*ds() # remove?
        a2 += div(J*inv(F)*U)*dx()
        #a2 += div(J*U*inv(F).T)*dx()

        # Eq. 31.2
        a2 += phi*q*fds # jacobian missing?
        a2 -= (P-Pext)*q*fds # p_{gamma}(t_n) - Pext

        # Eq. 31.3
        a2 += dt/rho_f*dot(dot(grad(phi),n), q)*nds  # Are Fs and stuff missing?
        a2 += dt/(rho_s*h_s)*phi*q*nds
        a2 -= dt/(rho_s*h_s)*phiext*q*nds
        a2 -= inner(Uext-(DFext-DFext1)/dt, n)*q*nds

        
        L2 = rhs(a2)
        a2 = lhs(a2)
        
        A2 = assemble(a2, keep_diagonal=True)
        b2 = assemble(L2)
        
        solver_p_corr = create_solver("bicgstab", "amg")
        #solver_p_corr = LinearSolver()

        # Velocity correction (u^n = tilde(u)^n+tau/rho*grad phi^n)
        a3 = inner(v, u)*dx()
        a3 -= inner(v, U)*dx()

        a3 += dt/rho_f*inner(v, grad(P - Pext))*dx()

        a3 += dt/rho_f*inner(v, grad(P-Pext))*dx()  # missing F?
        
        L3 = rhs(a3)
        a3 = lhs(a3)
        
        A3 = assemble(a3)
        b3 = assemble(L3)
        
        solver_u_corr = create_solver("gmres", "additive_schwarz")


        # Solid on boundary mesh (currently very simple model)
        # Is this solid sub-step eq. 1 p. 18?
        _dx = Measure("dx")

        a4 = rho_s*h_s/dt**2*inner(eta - 2*dgb1 + dgb2, x)*_dx # Second time-derivative of boundary displacement

        # Elastic operator
        a4 += inner(beta*eta, x)*_dx
        a4 += inner(beta1*grad(eta), grad(x))*_dx   # -lambda1*dxx(eta)

        # Viscous operator
        a4 += inner(alpha*rho_s*h_s*(eta - ETA1)/dt, x)*_dx
        a4 += inner(alpha1*beta1*grad((eta - ETA1)/dt), grad(x))*_dx

        a4 = rho_s*h_s/dt**2*inner(eta-2*dgb1+dgb2, x)*_dx # Second time-derivative of boundary displacement
        a4 += inner(beta*eta,x)*_dx
        #a4 += inner(PI(eta),x)*_dx
        a4 += inner(traction, x)*_dx # Force term

        L4 = rhs(a4)
        a4 = lhs(a4)

        A4 = assemble(a4)
        b4 = assemble(L4)
        solver_solid = create_solver("gmres", "hypre_euclid")

        # Mesh displacement (solve poisson equation with boundary displacement as bcs)
        a5 = inner(grad(d), grad(e))*dx()
        L5 = inner(Constant((0,)*dim), e)*dx()

        A5 = assemble(a5)
        b5 = assemble(L5)

        solver_displacement = create_solver("gmres", "hypre_euclid")    # same as solid solver. Seems to work
        #solver_displacement = create_solver("cg", "ml_amg")        # fails to return a valid solver. 

        # Helper functionality for getting boundary displacement as boundary condition        
        local_dofmapping = boundarymesh_to_mesh_dofmap(bmesh, Db, D)
        _keys, _values = zip(*local_dofmapping.iteritems())
        _keys = np.array(_keys, dtype=np.intc)
        _values = np.array(_values, dtype=np.intc)

        ofile = open("jconst.txt", "w")
        for timestep in xrange(start_timestep + 1, len(timesteps)):
            t.assign(timesteps[timestep])

            # Update various functions
            problem.update(spaces, U, P, t, timestep, bcs, None, None)
            timer.completed("problem update")
            
            # Solve tentative velocity
            assemble(a1, tensor=A1)
            assemble(L1, tensor=b1)

            for bc in bcu:
                bc.apply(A1, b1)
            
            #solve(A1, U.vector(), b1)
            solver_u_tent.solve(A1, U.vector(), b1)

            # Solve pressure correction
            #assemble(a2, tensor=A2, keep_diagonal=True)
            b2.apply('insert')
            assemble(a2, tensor=A2)
            assemble(L2, tensor=b2)

            for bc in bcp:
                bc.apply(A2, b2)
            
            #solve(A2, P.vector(), b2)
            b2.apply("insert")
            solver_p_corr.solve(A2, P.vector(), b2)
            
            #P-vector().zero()
            #P.vector().axpy(1.0, phi.vector())
            #P.vector().axpy(1.0, Pext.vector())

            # Solve updated velocity
            assemble(a3, tensor=A3)
            assemble(L3, tensor=b3)
            for bc in bcu:
                bc.apply(A3, b3)
            
            #solve(A3, U.vector(), b3)
            solver_u_corr.solve(A3, U.vector(), b3)

            # Solve solid
            traction.assign(T_bdry.compute(lambda x: {  "Velocity": U,
                                                        "Pressure": P,
                                                        "DynamicViscosity": mu,
                                                        "Displacement": DF}[x]
                            ))

            assemble(a4, tensor=A4)
            assemble(L4, tensor=b4)

            # missing BC for eta?, i.e. a4 or is it don ethrough b4?
            #for bc in bcs_shell:
            #    bc.apply(A4, b4)
            #embed()
            #sys.exit()
            #b4.array()[np.array([0, -1])] = 0
            
            #solve(A4, dgb.vector(), b4)
            solver_solid.solve(A4, dgb.vector(), b4)

            # Solve mesh displacement
            db.interpolate(dgb) # Interpolate to vertices on boundary
            get_set_vector(DF.vector(), _keys, db.vector(), _values) # Set boundary values to function

            for bc in bcs_eta:
                bc.apply(A5,b5)
                

            solver_displacement.solve(A5, DF.vector(), b5)

            # Rotate functions
            U1.assign(U)
            DF2.assign(DF1)
            DF1.assign(DF)
            dgb2.assign(dgb1)
            dgb1.assign(dgb)
            w.assign(DF - DF1)
            w.vector()[:] *= 1./float(dt) # Mesh velocity
            ETA1.assign(dgb)     # FIXME: Not sure about this 

            # Update extrapolations
            Uext.update(U)
            phiext.update(P - Pext)
            Pext.update(P)
            DFext2.assign(DFext1)
            DFext1.assign(DFext)
            DFext.update(DF)

            ofile.write("%s\n" % assemble(J*dx))

            yield ParamDict(spaces=spaces, observations=None, controls=None,
                    t=float(t), timestep=timestep, u=U, p=P,
                    d=DF, state=(U, P, DF))
