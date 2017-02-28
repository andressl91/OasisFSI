from dolfin import *
import numpy as np
from cbcpost import *

def before_first_compute(VQ,V1,Q,u,mesh):
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
