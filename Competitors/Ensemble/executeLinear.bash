#!/bin/bash
#set a job name  
#SBATCH --job-name=lin
#################  
#a file for job output, you can check job progress
#SBATCH --output=lin.out
#################
# a file for errors from the job
#SBATCH --error=lin.err
#################
#time you think you need; default is one day
#in minutes in this case, hh:mm:ss !!!!!!
#SBATCH --time=24:00:00
#################
#number of tasks you are requesting, SBATCH --exclusive
#SBATCH -n 5

#################
#partition to use
#SBATCH --partition=ioannidis
#SBATCH --mem=100Gb
#################
#number of nodes to distribute n tasks across
#################

python mainLinear.py test gif_pleasure $1 $2
