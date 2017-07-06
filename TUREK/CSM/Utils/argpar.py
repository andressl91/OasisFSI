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
    group.add_argument("-fluidvari",  help="Set variationalform for fluid                    --> Default=thetaCN", default="thetaCN")
    group.add_argument("-solidvari",  help="Set variationalform for solid                 --> Default=thetaCN", default="thetaCN")
    group.add_argument("-extravari",  help="Set variationalform for extrapolation                 --> Default=alfa", default="alfa")
    group.add_argument("-tag",  help="tag name for storing file                --> Default=alfa", default="tag")
    group.add_argument("-solver",   help="Set type of solver to be used            --> Default=newtonsolver", default="newtonsolver")
    group.add_argument("-p_deg",    type=int, help="Set degree of pressure                   --> Default=1", default=1)
    group.add_argument("-v_deg",    type=int, help="Set degree of velocity                   --> Default=2", default=2)
    group.add_argument("-d_deg",    type=int, help="Set degree of deformation                --> Default=2", default=2)
    group.add_argument("-theta",        type=float,  help="Theta Parameter for ex/imp/CN                             --> Default=1.0", default=1.0)
    group.add_argument("-extype",          help="Extrapolation constant (const, smallc, det)                            --> Default=const", default="const")
    group.add_argument("-bitype",          help="Different BC for extrapol (bc1, bc2)                            --> Default=default", default="bc1")
    group.add_argument("-T",        type=float,  help="Set end time                             --> Default=None", default=None)
    group.add_argument("-dt",       type=float,  help="Set timestep                             --> Default=None", default=None)
    group.add_argument("-r", "--refiner",   action="count", help="Mesh-refiner using built-in FEniCS method refine(Mesh)")
    return parser.parse_args()
