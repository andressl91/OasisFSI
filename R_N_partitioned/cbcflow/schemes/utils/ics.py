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

from __future__ import division

from cbcflow.dol import as_vector, project

# --- Initial condition helper functions for schemes

def assign_ics_mixed(up0, spaces, ics):
    """Assign initial conditions from ics to up0.

    up0 is a mixed function in spaces.W = spaces.V * spaces.Q,
    while ics = (icu, icp); icu = (icu0, icu1, ...).
    """
    up = as_vector(list(ics[0]) + [ics[1]])
    #project(up, spaces.W, function=up0) # TODO: Can do this in fenics dev
    upp = project(up, spaces.W)
    upp.rename("icup0_projection", "icup0_projection")
    up0.assign(upp)

def assign_ics_split(u0, p0, spaces, ics):
    """Assign initial conditions from ics to u0, p0.

    u0 is a vector valued function in spaces.V and p0 is a scalar function in spaces.Q,
    while ics = (icu, icp); icu = (icu0, icu1, ...).
    """
    u = as_vector(list(ics[0]))
    p = ics[1]
    u0.assign(project(u, spaces.V)) #, name="u0_init_projection"))
    p0.assign(project(p, spaces.Q)) #, name="p0_init_projection"))
    #project(u, spaces.V, function=u0) # TODO: Can do this in fenics dev
    #project(p, spaces.Q, function=p0) # TODO: Can do this in fenics dev

def assign_ics_segregated(u0, p0, spaces, ics):
    """Assign initial conditions from ics to u0[:], p0.

    u0 is a list of scalar functions each in spaces.U and p0 is a scalar function in spaces.Q,
    while ics = (icu, icp); icu = (icu0, icu1, ...).
    """
    for d in spaces.dims:
        u0[d].assign(project(ics[0][d], spaces.U)) #, name="u0_%d_init_projection"%d))
        #project(ics[0][d], spaces.U, function=u0[d]) # TODO: Can do this in fenics dev
    p0.assign(project(ics[1], spaces.Q)) #, name="p0_init_projection"))
    #project(ics[1], spaces.Q, function=p0) # TODO: Can do this in fenics dev
