
from cbcpost import *
from cbcpost.utils import *
from cbcflow import *
from cbcflow.schemes.utils import *
from dolfin import *
import numpy as np
from IPython import embed

class FSIProblem(NSProblem):
    @classmethod
    def default_params(cls):
        """Returns the default parameters for an FSI problem.
        (Extends NSProblem.default_params())

        Explanation of parameters:

        Physical parameters:

          - E: float, kinematic viscosity
          - rho_s: float, mass density
          - R: float, Radius to use
          - h: float, Wall thickness
          - nu: float, Poisson ratio

        """

        params = NSProblem.default_params()
        params.update(
            E=1.0,
            rho_s=1.0,
            R = 1.0,
            h = 1.0,
            nu = 0.3,
        )

        return params


class BoundaryTraction(Field):
    def add_fields(self):
        return [DynamicViscosity()]

    def before_first_compute(self, get):
        u = get("Velocity")
        V = u.function_space()

        spaces = SpacePool(V.mesh())

        Q = spaces.get_space(0,1)
        Q_boundary = spaces.get_space(Q.ufl_element().degree(), 1, boundary=True)

        self.v = TestFunction(Q)
        self.traction = Function(Q, name="BoundaryTraction_full")
        self.traction_boundary = Function(Q_boundary, name="BoundaryTraction")

        local_dofmapping = mesh_to_boundarymesh_dofmap(spaces.BoundaryMesh, Q, Q_boundary)
        self._keys = np.array(local_dofmapping.keys(), dtype=np.intc)
        self._values = np.array(local_dofmapping.values(), dtype=np.intc)
        self._temp_array = np.zeros(len(self._keys), dtype=np.float_)

        _dx = Measure("dx")
        Mb = assemble(inner(TestFunction(Q_boundary), TrialFunction(Q_boundary))*_dx)
        self.solver = create_solver("gmres", "jacobi")
        self.solver.set_operator(Mb)

        self.b = Function(Q_boundary).vector()

        self._n = FacetNormal(V.mesh())
        self.I = SpatialCoordinate(V.mesh())

    def compute(self, get):
        u = get("Velocity")
        p = get("Pressure")
        mu = get("DynamicViscosity")
        d = get("Displacement")

        if isinstance(mu, (float, int)):
            mu = Constant(mu)

        A = self.I+d # ALE map
        F = grad(A)
        J = det(F)

        n = self._n
        S = J*sigma(mu, u, p)*inv(F).T*n

        form = inner(self.v, S)*ds()
        assemble(form, tensor=self.traction.vector())

        get_set_vector(self.b, self._keys, self.traction.vector(), self._values, self._temp_array)

        # Ensure proper scaling
        self.solver.solve(self.traction_boundary.vector(), self.b)

        # Tests
        _dx = Measure("dx")
        File("trac.pvd") << self.traction_boundary

        return self.traction_boundary


def epsilon(u):
    return 0.5*(grad(u)+grad(u).T)

def Epsilon(u, F):
    return 0.5*(grad(u)*inv(F)+inv(F).T*grad(u).T)  # * or dot?

def sigma(mu, u,p):
    return -p*Identity(u.cell().topological_dimension())+2*mu*epsilon(u)

def Sigma(mu,u,p,F):
    return -p*Identity(u.cell().topological_dimension())+2*mu*Epsilon(u, F)


class AbsorbingStress(Constant):
    "Implemented from Nobile and Vergara paper"
    def __init__(self, problem, facet_domains, indicator):
        Constant.__init__(self, 0)
        #self.facet_domains = facet_domains
        #self.indicator = indicator
        self.ds = Measure("ds")[facet_domains](indicator)
        self.ds = self.ds.reconstruct(domain=facet_domains.mesh())
        self.A0 = assemble(Constant(1)*self.ds)
        self.beta = problem.params.E*problem.params.h/(1-problem.params.nu**2)*1.0/problem.params.R**2
        self.rho_f = problem.params.rho
        self.n = FacetNormal(facet_domains.mesh())

    def update(self, u):
        An = assemble(Constant(1)*self.ds)
        Fn = assemble(dot(u, self.n)*self.ds)
        val = ((sqrt(self.rho_f)/(2*sqrt(2))*Fn/An+sqrt(self.beta*sqrt(self.A0)))**2-self.beta*sqrt(self.A0))
        self.assign(val)
