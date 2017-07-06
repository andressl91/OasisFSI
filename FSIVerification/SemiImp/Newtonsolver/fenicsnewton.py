from dolfin import *

def solver_setup(F_fluid_linear, F_fluid_nonlinear, \
                 F_solid_linear, F_solid_nonlinear, DVP, dvp_, bcs, **monolithic):

    class FluidCoupled(NonlinearProblem):
        def __init__(self, a, L, bcs):
            NonlinearProblem.__init__(self)
            self.L = L
            self.a = a
            self.bcs = bcs
        def F(self, b, x):
            #[bc.apply(b) for bc in bcs]
            assemble(self.L, tensor=b)
            [bc.apply(b, x) for bc in self.bcs]
        def J(self, A, x):
            assemble(self.a, tensor=A).ident_zeros()
            [bc.apply(A) for bc in self.bcs]


    F_lin = F_fluid_linear + F_solid_linear
    F_nonlin = F_fluid_nonlinear + F_solid_nonlinear
    F = F_lin + F_nonlin

    chi = TrialFunction(DVP)
    Jac = derivative(F, dvp_["n"], chi)

    problem = FluidCoupled(Jac, F, bcs)
    newton_solver = NewtonSolver()
    newton_solver.parameters["linear_solver"] = "mumps"
    newton_solver.parameters["convergence_criterion"] = "incremental"
    newton_solver.parameters["relative_tolerance"] = 1e-6


    return dict(F=F, Jac=Jac, newton_solver=newton_solver, problem=problem)


def newtonsolver(F, dvp_, t, newton_solver, problem, **monolithic):

    newton_solver.solve(problem, dvp_["n"].vector())
    return dict(t=t)
