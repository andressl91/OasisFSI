import argparse
from argparse import RawTextHelpFormatter

def parse():
    parser = argparse.ArgumentParser(description="Implementation of Turek test case CFD1\n"
    "For details: http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.550.1689&rep=rep1&type=pdf",\
     formatter_class=RawTextHelpFormatter, \
      epilog="############################################################################\n"
      "Example --> python fsi1.py\n"
      "############################################################################")
    group = parser.add_argument_group('Parameters')
    group.add_argument("-p_deg",  type=int, help="Set degree of pressure                     --> Default=1", default=1)
    group.add_argument("-v_deg",  type=int, help="Set degree of velocity                     --> Default=2", default=2)
    group.add_argument("-d_deg",  type=int, help="Set degree of velocity                     --> Default=1", default=1)
    group.add_argument("-T",  type=float, help="Set end time                                 --> Default=0.1", default=0.1)
    group.add_argument("-dt",  type=float, help="Set timestep                                --> Default=0.001", default=0.001)
    group.add_argument("-theta",  type=float, help="Explicit, Implicit, Cranc-Nic (0, 1, 0.5)  --> Default=1", default=1)
    group.add_argument("-discr",  help="Write out or keep tensor in variational form --> Default=keep", default="keep")
    group.add_argument("-r", "--refiner", action="count", help="Mesh-refiner using built-in FEniCS method refine(Mesh)")
    return parser.parse_args()
