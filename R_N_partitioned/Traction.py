# From IBCS
#u = get("Velocity")
import numpy as np

from dolfin import *
from cbcpost import *
from cbcpost.utils import *
from mappings import *

def boundary_compute(u_,d_, p_):
    V = u_.function_space()

    spaces = SpacePool(V.mesh()) #Think this pull out all the spaces from the mesh

    Q = spaces.get_space(0,1) #Think this pulls out the V1 space
    Q_boundary = spaces.get_space(Q.ufl_element().degree(), 1, boundary=True) # Taking out only the boundary and making a space

    v = TestFunction(Q)
    traction = Function(Q, name="BoundaryTraction_full")
    traction_boundary = Function(Q_boundary, name="BoundaryTraction") #boundary function

    local_dofmapping = mesh_to_boundarymesh_dofmap(spaces.BoundaryMesh, Q, Q_boundary) #getting dofmap from mixed to single space

    _keys = np.array(local_dofmapping.keys(), dtype=np.intc)
    _values = np.array(local_dofmapping.values(), dtype=np.intc)
    _temp_array = np.zeros(len(_keys), dtype=np.float_)

    _dx = Measure("dx")
    Mb = assemble(inner(TestFunction(Q_boundary), TrialFunction(Q_boundary))*_dx) #not sure think this sets up the size of the operator for solve later
    pc = PETScPreconditioner("jacobi")
    solver = PETScKrylovSolver("gmres",pc)
    #solver = create_solver("gmres", "jacobi") # from IBCS
    solver.set_operator(Mb)

    b = Function(Q_boundary).vector()

    _n = FacetNormal(V.mesh())
    I = SpatialCoordinate(V.mesh())
    #return b,solver,I,_n,v,traction,_keys,_values,_temp_array,traction_boundary

    #def boundary_compute(b,solver,I,_n,v,traction,traction_boundary,_keys,_values,_temp_array,d_,u_,p_):

    A = I+d_("-") # ALE map
    F = grad(A)
    J = det(F)

    S = J*sigma_f(u_("-"),p_("-"))*inv(F).T*_n("-")

    form = inner(v("-"), S)*dS(5)
    assemble(form, tensor=traction.vector()) #assembles a testfunction agains the force on the interface boundary

    get_set_vector(b, _keys, traction.vector(), _values, _temp_array) # not sure, think this sets up the vector b in the right space from mixed space
    #miros way : get_set_vector(self.b, self._keys, self.traction.vector(), self._values, self._temp_array)

    # Ensure proper scaling
    solver.solve(traction_boundary.vector(), b) # gives the traction_boundary function values from b on the interface boundary

    return traction_boundary
