import argparse
from argparse import RawTextHelpFormatter

parser = argparse.ArgumentParser(description="Implementation of Turek test case FSI\n"
"For details: http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.550.1689&rep=rep1&type=pdf",\
 formatter_class=RawTextHelpFormatter, \
  epilog="############################################################################\n"
  "Example --> python ALE_FSI.py \n"
  "Example --> python FSI_ALE_Partitioned.py -v_deg 1 -p_deg 1 -d_deg 1 -dt 0.5 -T 10 -step 1 -FSI_number 1  (Refines mesh one time, -rr for two etc.) \n"
  "############################################################################")
group = parser.add_argument_group('Parameters')
group.add_argument("-p_deg",       type=int,   help="Set degree of pressure                     --> Default=1", default=1)
group.add_argument("-v_deg",       type=int,   help="Set degree of velocity                     --> Default=2", default=1)
group.add_argument("-d_deg",       type=int,   help="Set degree of displacement                 --> Default=2", default=1)
group.add_argument("-FSI_number",  type=int,   help="FSI number                                 --> Default=1", default=1)
group.add_argument("-T",           type=float, help="End time                     --> Default=20", default=20)
group.add_argument("-dt",          type=float, help="Time step                     --> Default=0.5", default=0.5)
group.add_argument("-step",          type=float, help="savestep                     --> Default=1", default=1)
group.add_argument("-r", "--refiner", action="count", help="Mesh-refiner using built-in FEniCS method refine(Mesh)")
group.add_argument("-beta",          type=float, help="AC factor                     --> Default=0.5", default=0.5)
args = parser.parse_args()

v_deg = args.v_deg
p_deg = args.p_deg
d_deg = args.d_deg
T = args.T
dt = args.dt
beta = args.beta
#ref = args.r
step = args.step
fig = False
FSI_deg = args.FSI_number
