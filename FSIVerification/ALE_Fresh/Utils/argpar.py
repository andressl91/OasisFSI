import argparse
from argparse import RawTextHelpFormatter

def parse():
    parser = argparse.ArgumentParser(description="Monolithic FSI solver\n",\
     formatter_class=RawTextHelpFormatter, \
      epilog="############################################################################\n"
      "Example --> python monolithic.py -problem fsi1 -T 10 -dt 0.5\n"
      "############################################################################")
    group = parser.add_argument_group('Parameters')
    group.add_argument("-problem",  help="Set problem to solve                     --> Default=fsi1", default="fsi1")
    group.add_argument("-solver",   help="Set type of solver to be used            --> Default=fsi1", default="newtonsolver")
    group.add_argument("-p_deg",    type=int, help="Set degree of pressure                   --> Default=1", default=1)
    group.add_argument("-v_deg",    type=int, help="Set degree of velocity                   --> Default=2", default=2)
    group.add_argument("-d_deg",    type=int, help="Set degree of deformation                --> Default=1", default=2)
    group.add_argument("-T",        type=float,  help="Set end time                             --> Default=0.1", default=1)
    group.add_argument("-dt",       type=float,  help="Set timestep                             --> Default=0.001", default=0.5)
    group.add_argument("-r", "--refiner",   action="count", help="Mesh-refiner using built-in FEniCS method refine(Mesh)")
    return parser.parse_args()
