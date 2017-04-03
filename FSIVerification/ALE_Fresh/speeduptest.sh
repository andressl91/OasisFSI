#!/bin/bash

prob=(naive reusejac reducequad newtonsolver newtonsolver_cheapjacobi)
#prob=(fenicsnewton naive)
ELEMPROB=${#prob[@]}

for (( i=0;i<$ELEMPROB;i++)); do
    	mpirun --np 2 python monolithic.py -dt 0.5 -T 1 -problem speedup -solver ${prob[${i}]}
done
