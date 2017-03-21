import argparse
from argparse import RawTextHelpFormatter

def parse():
    parser = argparse.ArgumentParser(description="Monolithic FSI solver\n",\
     formatter_class=RawTextHelpFormatter, \
      epilog="############################################################################\n"
      "Example --> Something\n"
      "############################################################################")
    group = parser.add_argument_group('Parameters')
    group.add_argument("-problem",  help="Set degree of pressure                               --> Default=fsi1", default="fsi1")
    group.add_argument("-p_deg",    type=int, help="Set degree of pressure                     --> Default=1", default=1)
    group.add_argument("-v_deg",    type=int, help="Set degree of velocity                     --> Default=2", default=2)
    group.add_argument("-d_deg",    type=int, help="Set degree of velocity                     --> Default=1", default=1)
    group.add_argument("-T",        type=float,   help="Set end time                           --> Default=0.1", default=0.1)
    group.add_argument("-dt",       type=float,  help="Set timestep                            --> Default=0.001", default=0.001)
    group.add_argument("-r", "--refiner",   action="count", help="Mesh-refiner using built-in FEniCS method refine(Mesh)")
    group.add_argument("-l", "--list", action="store_true")
    return parser.parse_args()
