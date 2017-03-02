# Copyright (C) 2010-2014 Simula Research Laboratory
#
# This file is part of CBCFLOW.
#
# CBCFLOW is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# CBCFLOW is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with CBCFLOW. If not, see <http://www.gnu.org/licenses/>.


from cbcflow.dol import (FunctionSpace, VectorFunctionSpace,
                         TensorFunctionSpace, BoundaryMesh, dolfin_version,
                         MixedElement)
from cbcpost import SpacePool
from distutils.version import LooseVersion

def galerkin_family(degree):
    return "CG" if degree > 0 else "DG"

def decide_family(family, degree):
    return galerkin_family(degree) if family == "auto" else family

class NSSpacePool():
    "A function space pool with custom named spaces for use with Navier-Stokes schemes."
    def __init__(self, mesh, u_degree, p_degree, u_family="auto", p_family="auto"):
        self.spacepool = SpacePool(mesh)
        assert isinstance(u_degree, int)
        assert isinstance(p_degree, int)
        assert isinstance(u_family, str)
        assert isinstance(p_family, str)
        self.u_degree = u_degree
        self.p_degree = p_degree
        self.u_family = u_family
        self.p_family = p_family
        self._spaces = {}

        # Get dimensions for convenience
        cell = mesh.ufl_cell()
        self.gdim = cell.geometric_dimension()
        self.tdim = cell.topological_dimension()
        self.gdims = range(self.gdim)
        self.tdims = range(self.tdim)

        # For compatibility, remove when code has been converted
        self.d = self.gdim
        self.dims = self.gdims

    @property
    def U(self):
        "Scalar valued space for velocity components."
        return self.spacepool.get_space(self.u_degree, 0, family=self.u_family)

    @property
    def V(self):
        "Vector valued space for velocity vector."
        return self.spacepool.get_space(self.u_degree, 1, family=self.u_family)

    @property
    def Q(self):
        "Scalar valued space for pressure."
        return self.spacepool.get_space(self.p_degree, 0, family=self.p_family)

    @property
    def DU0(self):
        "Scalar valued space for gradient component of single velocity component."
        return self.spacepool.get_space(self.u_degree-1, 0, family=self.u_family)

    @property
    def DU(self):
        "Vector valued space for gradients of single velocity components."
        return self.spacepool.get_space(self.u_degree-1, 1, family=self.u_family)

    @property
    def DV(self):
        "Tensor valued space for gradients of velocity vector."
        return self.spacepool.get_space(self.u_degree-1, 2, family=self.u_family)

    @property
    def DQ0(self):
        "Scalar valued space for pressure gradient component."
        return self.spacepool.get_space(self.p_degree-1, 0, family=self.p_family)

    @property
    def DQ(self):
        "Vector valued space for pressure gradient."
        return self.spacepool.get_space(self.p_degree-1, 1, family=self.p_family)

    @property
    def W(self):
        "Mixed velocity-pressure space."
        space = self._spaces.get("W")
        if space is None:
            if LooseVersion(dolfin_version()) > LooseVersion("1.6.0"):
                space = FunctionSpace(self.spacepool.mesh, MixedElement(self.V.ufl_element(),self.Q.ufl_element()))
            else:
                space = self.V*self.Q
            self._spaces["W"] = space
        return space

class NSSpacePoolMixed(NSSpacePool):
    "A function space pool with custom named spaces for use with mixed Navier-Stokes schemes."
    pass

class NSSpacePoolSplit(NSSpacePool):
    "A function space pool with custom named spaces for use with split Navier-Stokes schemes."
    pass

class NSSpacePoolSegregated(NSSpacePool):
    "A function space pool with custom named spaces for use with segregated Navier-Stokes schemes."
    pass
