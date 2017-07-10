#!/bin/bash

mpirun -np 4 python csm.py -T 8 -dt 0.02 -problem csm3 -theta 1.0 -solver reusejac2 -v_deg 3 -d_deg 3
mpirun -np 4 python csm.py -T 8 -dt 0.02 -problem csm3_b1 -theta 1.0 -solver reusejac2 -v_deg 3 -d_deg 3
mpirun -np 4 python csm.py -T 8 -dt 0.02 -problem csm3_b2 -theta 1.0 -solver reusejac2 -v_deg 3 -d_deg 3

mpirun -np 4 python csm.py -T 8 -dt 0.01 -problem csm3 -theta 1.0 -solver reusejac2 -v_deg 3 -d_deg 3
mpirun -np 4 python csm.py -T 8 -dt 0.01 -problem csm3_b1 -theta 1.0 -solver reusejac2 -v_deg 3 -d_deg 3
mpirun -np 4 python csm.py -T 8 -dt 0.01 -problem csm3_b2 -theta 1.0 -solver reusejac2 -v_deg 3 -d_deg 3

mpirun -np 4 python csm.py -T 8 -dt 0.005 -problem csm3 -theta 1.0 -solver reusejac2 -v_deg 3 -d_deg 3
mpirun -np 4 python csm.py -T 8 -dt 0.005 -problem csm3_b1 -theta 1.0 -solver reusejac2 -v_deg 3 -d_deg 3
mpirun -np 4 python csm.py -T 8 -dt 0.005 -problem csm3_b2 -theta 1.0 -solver reusejac2 -v_deg 3 -d_deg 3
