#!/bin/bash
#SBATCH -A dfc13_mri
#SBATCH -p mgc-open
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=16GB
#SBATCH --time=96:00:00
#SBATCH --job-name=seti_overlap
#SBATCH --chdir=/storage/home/mlp95/work/seti-fortuitous-obs
#SBATCH --output=/storage/home/mlp95/work/logs/seti_overlap.%j.out

echo "About to start: $SLURM_JOB_NAME"
date
echo "Job id: $SLURM_JOBID"
echo "About to change into $SLURM_SUBMIT_DIR"
cd $SLURM_SUBMIT_DIR
echo "About to start Python"
source /storage/group/ebf11/default/software/anaconda3/bin/activate
conda activate seti
python process_archive.py
echo "Python exited"
date
