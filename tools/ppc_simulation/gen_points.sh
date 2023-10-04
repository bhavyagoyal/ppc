#!/usr/bin/env bash

#SBATCH --array=1-1%4

#SBATCH --partition=research
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=24:0:0

###SBATCH --exclude=euler[01-23],euler[28-30]
###SBATCH --exclude=euler[01-23]

###SBATCH -o slurm.%j.%N.out # STDOUT
###SBATCH -e slurm.%j.%N.err # STDERR
#SBATCH -o logs/slurm.%A.%a.%N.out
#SBATCH -e logs/slurm.%A.%a.%N.err

#SBATCH --job-name=mmdsim
#SBATCH --no-requeue

module load conda/miniforge/23.1.0
bootstrap_conda
conda activate openmmlab2

#START=$1
#END=$2
START=$((($SLURM_ARRAY_TASK_ID-1)*100))
END=$((($SLURM_ARRAY_TASK_ID)*100))

DATASET="sunrgbd"
#DATASET="kitti"
THRESHOLD=0.3

if [ $DATASET == "kitti" ]; then
	SBR=("5_100" "5_250" "5_500" "5_1000")
else
	SBR=("5_50" "5_100" "1_50" "1_100")
fi

for i in "${!SBR[@]}"
do
python -u gen_points.py --method=argmax-filtering-sbr --sbr=${SBR[$i]} --start $START --end $END --threshold $THRESHOLD --outfolder_prefix "${THRESHOLD}" --dataset "${DATASET}" 

done


