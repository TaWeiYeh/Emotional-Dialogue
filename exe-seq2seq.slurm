#!/bin/bash

##NECESSARY JOB SPECIFICATIONS
#SBATCH --job-name=Seq2seq           #Set the job name to "JobExample4"
#SBATCH --time=02:00:00              #Set the wall clock limit 
#SBATCH --ntasks=1                   #Request 1 task
#SBATCH --mem=32G                    #Request 32.5GB per node
#SBATCH --output=./log/Seq2seqOut.%j #Send stdout/err to "Example4Out.[jobID]"
#SBATCH --gres=gpu:1                 #Request 1 GPU per node can be 1 or 2
#SBATCH --partition=gpu              #Request the GPU partition/queue

##OPTIONAL JOB SPECIFICATIONS
##SBATCH --account=123456             #Set billing account to 123456
##SBATCH --mail-type=ALL              #Send email on all job events
##SBATCH --mail-user=twy359@tamu.edu    #Send all emails to email_address 

##First Executable Line
cd /scratch/user/twy359/codes/Emotional-Dialogue/
. load_modules

python train_seq2seq.py
