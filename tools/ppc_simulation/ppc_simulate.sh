#!/usr/bin/env bash

#SBATCH --array=1-104%10

#SBATCH --partition=research
#SBATCH --cpus-per-task=8
#SBATCH --mem=50G
#SBATCH --time=24:0:0

#SBATCH --exclude=euler[01-23],euler[28-30]

###SBATCH -o slurm.%j.%N.out # STDOUT
###SBATCH -e slurm.%j.%N.err # STDERR
#SBATCH -o logs/slurm.%A.%a.%N.out
#SBATCH -e logs/slurm.%A.%a.%N.err

#SBATCH --job-name=pensim
#SBATCH --no-requeue


module load matlab/r2021b

DATASET="sunrgbd"
#DATASET="kitti"
START=$1
END=$2
#START=$((($SLURM_ARRAY_TASK_ID-1)*100))
#END=$((($SLURM_ARRAY_TASK_ID)*100))

matlab -nodisplay -nosplash -nodesktop -r 'SimulatePPCMeasurements('$((START+1))','$END',"'$DATASET'");quit;'


