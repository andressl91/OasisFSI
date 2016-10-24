#!/bin/bash
#Job name:
#$SBATCH --job-name=taylorgreen
#
#Project:
#SBATCH --account=uio
#
#Wall clock limit:
#SBATCH --time='00:05:00'
#
#Max memory usage per task:
#SBATCH --mem-per-cpu=6000M
#
#Number of tasks(cores):
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=1

##Set up job enviroment
source /cluster/bin/jobsetup

echo $1 $2

#module load gcc/4.9.2
#module load openmpi.gnu/1.8.4
source ~oyvinev/fenics1.6/fenics1.6

#Expand pythonpath with locally installed packages
export PYTHONPATH=$PYTHONPATH:$HOME/.local/lib/python2.7/site-packages/

#Define what to do after work is finished
cleanup "mkdir -p $HOME/results"
##cleanup "cp -r $SCRATCH/* $HOME/results"
##cleanup "cp -r $SCRATCH/results/Drag.txt $HOME/OasisFSI/FluidVerification/Turek_Bar/results"
cleanup "cp -r $SCRATCH/results $HOME/results"

echo "SCRATCH is $SCRATCH"
cp fluid_validation.py turek2.xml -r $HOME/OasisFSI/FluidVerification/Turek_Bar/results $SCRATCH
cd $SCRATCH
ls
echo $SCRATCH


# get number of elements in the array
#ELEMDT=${#DT[@]}
#ELEMN=${#N[@]}

#for (( i=0;i<$ELEMN;i++)); do
#    for (( j=0;j<$ELEMDT;j++)); do
#    	mpirun --bind-to none python NSfracStep.py problem=TaylorGreen3D T=1.0 dt =0.001 Nx=64 Ny=64 Nx=64 nu=0.001 
#    done
#done

mpirun --bind-to none python fluid_validation.py -T 0.01 -dt 0.001 -v_deg 1 -solver Newton -fig
