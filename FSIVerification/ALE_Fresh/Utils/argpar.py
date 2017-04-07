import argparse
from argparse import RawTextHelpFormatter

def parse():
    parser = argparse.ArgumentParser(description="Monolithic FSI solver\n",\
     formatter_class=RawTextHelpFormatter, \
      epilog="############################################################################\n"
      "Example --> python monolithic.py -problem fsi1 -T 10 -dt 0.5\n"
      "CSM example python monolithic.py -fluidvari nofluid -solidvari csm -extravari alfa -solver nofluid -problem csm1 -T 10 -dt 0.2 -theta 1"
      "############################################################################")
    group = parser.add_argument_group('Parameters')
    group.add_argument("-problem",  help="Set problem to solve                     --> Default=fsi1", default="fsi1")
    group.add_argument("-fluidvari",  help="Set variationalform for fluid                    --> Default=fluid_coupled", default="fluid_coupled")
    group.add_argument("-solidvari",  help="Set variationalform for solid                 --> Default=CN_mixed", default="CN_mixed")
    group.add_argument("-extravari",  help="Set variationalform for extrapolation                 --> Default=alfa", default="alfa")
    group.add_argument("-solver",   help="Set type of solver to be used            --> Default=newtonsolver", default="newtonsolver")
    group.add_argument("-p_deg",    type=int, help="Set degree of pressure                   --> Default=1", default=1)
    group.add_argument("-v_deg",    type=int, help="Set degree of velocity                   --> Default=2", default=2)
    group.add_argument("-d_deg",    type=int, help="Set degree of deformation                --> Default=1", default=2)
    group.add_argument("-theta",        type=float,  help="Skew Cranc-Nic                             --> Default=None", default=1/2.)
    group.add_argument("-extype",          help="Extrapolation constant (const, smallc, det)                            --> Default=default", default="const")
    group.add_argument("-T",        type=float,  help="Set end time                             --> Default=None", default=None)
    group.add_argument("-dt",       type=float,  help="Set timestep                             --> Default=None", default=None)
    group.add_argument("-r", "--refiner",   action="count", help="Mesh-refiner using built-in FEniCS method refine(Mesh)")
    return parser.parse_args()
