from dolfin import *
from cbcpost import *
from cbcflow import *
from pipe2d import Pipe2D
from fsi_decoupled import FSI_Decoupled
import sys
from IPython import embed

if __name__ == '__main__':
    set_log_level(100)

    problem = Pipe2D(dict(dt=1e-3, N=10, stress_amplitude=1e2, stress_time=1.0, T=1,
        meshpath="meshes/pipe/pipe_4k.h5"))


    from fsi_decoupled import FSI_Decoupled
    scheme = FSI_Decoupled(dict(r=1, s=0))

    _plot = True
    pp = PostProcessor(dict(casedir="Results_" + problem.__class__.__name__, clean_casedir=True))
    pp.add_field(SolutionField("Velocity", dict(save=True, plot=_plot, save_as="xdmf")))
    pp.add_field(SolutionField("Pressure", dict(save=True, plot=_plot,save_as="xdmf")))
    pp.add_field(SolutionField("Displacement", dict(save=True, plot=False, save_as="xdmf")))

    class MassConservation(Field):
        def compute(self, get):
            u = get("Velocity")
            n = FacetNormal(u.function_space().mesh())
            return assemble(dot(u,n)*ds())

    _plot = True
    pp.add_field(MassConservation(dict(plot=_plot)))

    solver = NSSolver(problem, scheme, pp)
    solver.solve()
    exit()
