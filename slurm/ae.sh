#!/bin/bash
#SBATCH --mem=40GB
#SBATCH --output=/fsx-llm/pbelcak/pytorch-cifar/logs/%j.out
#SBATCH --error=/fsx-llm/pbelcak/pytorch-cifar/logs/%j.err
#SBATCH --mail-type=NONE                            # mail configuration: NONE, BEGIN, END, FAIL, REQUEUE, ALL
#SBATCH --cpus-per-task=2                         	# number of CPUs per task

# Exit on errors
set -o errexit

module avail
module load cuda/11.6 \
    nccl/2.12.7-cuda.11.6 \
    nccl_efa/1.15.1-nccl.2.12.7-cuda.11.6

PROJECT_NAME=pytorch-cifar	# if you're changing this, remember to change the destination of the logs (header of this file) as well
USER=pbelcak				# if you're changing this, remember to change the user of the logs (header of this file) as well
PROJECT_PATH=/data/home/${USER}/${PROJECT_NAME}
STORAGE_PATH=/fsx-llm/${USER}/${PROJECT_NAME}

ARCHITECTURE=${1:-vgg19_full}

export NCCL_DEBUG=info

# Binary or script to execute
PYTHONPATH=${PROJECT_PATH} python ${PROJECT_PATH}/main.py \
	--job_id=${SLURM_JOB_ID} \
	--job_suite=${ARCHITECTURE} \
	--seed=0 \
	--data_directory=${STORAGE_PATH}/data/ \
	--checkpointing_directory=${STORAGE_PATH}/checkpoints/ \
	--logging_directory=${STORAGE_PATH}/logs/ \
	--results_directory=${STORAGE_PATH}/results/ \
	--action=train \
	--dataset=cifar10 \
	--architecture=${ARCHITECTURE} \
	--optimizer=sgd \
	--scheduler=cosine \
	--learning_rate=0.1 \
	--epochs=100 \
	--patience=100 \
	--min_delta=0.01 \
	--batch_size=2048 \
	--clip=1.0 \
	--fixation_schedule=linear_100 \
	--evaluate_after_training ${@:2}

# Send more noteworthy information to the output log
echo "Finished at:     $(date)"

# End the script with exit code 0
exit 0