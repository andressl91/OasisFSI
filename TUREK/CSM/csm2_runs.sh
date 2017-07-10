#!/bin/bash

python csm.py -T 8 -dt 0.1 -problem csm2 -theta 1.0 -solver reusejac2 -v_deg 1 -d_deg 1
python csm.py -T 8 -dt 0.1 -problem csm2 -theta 1.0 -solver reusejac2 -v_deg 2 -d_deg 2
python csm.py -T 8 -dt 0.1 -problem csm2 -theta 1.0 -solver reusejac2 -v_deg 3 -d_deg 3

python csm.py -T 8 -dt 0.05 -problem csm2 -theta 1.0 -solver reusejac2 -v_deg 1 -d_deg 1
python csm.py -T 8 -dt 0.05 -problem csm2 -theta 1.0 -solver reusejac2 -v_deg 2 -d_deg 2
python csm.py -T 8 -dt 0.05 -problem csm2 -theta 1.0 -solver reusejac2 -v_deg 3 -d_deg 3

mpirun -np 4 python csm.py -T 8 -dt 0.1 -problem csm2_b1 -theta 1.0 -solver reusejac2 -v_deg 1 -d_deg 1
mpirun -np 4 python csm.py -T 8 -dt 0.1 -problem csm2_b1 -theta 1.0 -solver reusejac2 -v_deg 2 -d_deg 2
mpirun -np 4 python csm.py -T 8 -dt 0.1 -problem csm2_b1 -theta 1.0 -solver reusejac2 -v_deg 3 -d_deg 3

mpirun -np 4 python csm.py -T 8 -dt 0.05 -problem csm2_b1 -theta 1.0 -solver reusejac2 -v_deg 1 -d_deg 1
mpirun -np 4 python csm.py -T 8 -dt 0.05 -problem csm2_b1 -theta 1.0 -solver reusejac2 -v_deg 2 -d_deg 2
mpirun -np 4 python csm.py -T 8 -dt 0.05 -problem csm2_b1 -theta 1.0 -solver reusejac2 -v_deg 3 -d_deg 3

mpirun -np 4 python csm.py -T 8 -dt 0.1 -problem csm2_b2 -theta 1.0 -solver reusejac2 -v_deg 1 -d_deg 1
mpirun -np 4 python csm.py -T 8 -dt 0.1 -problem csm2_b2 -theta 1.0 -solver reusejac2 -v_deg 2 -d_deg 2
mpirun -np 4 python csm.py -T 8 -dt 0.1 -problem csm2_b2 -theta 1.0 -solver reusejac2 -v_deg 3 -d_deg 3

mpirun -np 4 python csm.py -T 8 -dt 0.05 -problem csm2_b2 -theta 1.0 -solver reusejac2 -v_deg 1 -d_deg 1
mpirun -np 4 python csm.py -T 8 -dt 0.05 -problem csm2_b2 -theta 1.0 -solver reusejac2 -v_deg 2 -d_deg 2
mpirun -np 4 python csm.py -T 8 -dt 0.05 -problem csm2_b2 -theta 1.0 -solver reusejac2 -v_deg 3 -d_deg 3
