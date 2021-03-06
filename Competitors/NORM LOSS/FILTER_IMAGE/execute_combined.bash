#!/bin/bash
#set a job name  
#SBATCH --job-name=im_filt
#################  
#a file for job output, you can check job progress
#SBATCH --output=im_filt.out
#################
# a file for errors from the job
#SBATCH --error=im_filt.err
#################
#time you think you need; default is one day
#in minutes in this case, hh:mm:ss
#SBATCH --time=24:00:00
#################
#number of tasks you are requesting
#SBATCH -N 1
#SBATCH --exclusive
#################
#partition to use
#SBATCH --partition=gpu
#SBATCH --mem=100Gb
#################
#number of nodes to distribute n tasks across
#################

python main_combined_network.py train $1 $2 $3
