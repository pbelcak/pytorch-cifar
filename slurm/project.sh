#!/bin/bash
#SBATCH --mem=60GB
#SBATCH --output=/home/pbelcak/pytorch-cifar/log/%j.out
#SBATCH --error=/home/pbelcak/pytorch-cifar/log/%j.err
#SBATCH --mail-type=NONE                            # mail configuration: NONE, BEGIN, END, FAIL, REQUEUE, ALL
#SBATCH --cpus-per-task=2                         	# number of CPUs per task

# Exit on errors
set -o errexit

# Set a directory for temporary files unique to the job with automatic removal at job termination
TMPDIR=$(mktemp -d)
if [[ ! -d ${TMPDIR} ]]; then
	echo 'Failed to create temp directory' >&2
	exit 1
fi
trap "exit 1" HUP INT TERM
trap 'rm -rf "${TMPDIR}"' EXIT
export TMPDIR

# Change the current directory to the location where you want to store temporary files, exit if changing didn't succeed.
# Adapt this to your personal preference
cd "${TMPDIR}" || exit 1

# Send some noteworthy information to the output log
echo "Running on node: $(hostname)"
echo "In directory:    $(pwd)"
echo "Starting on:     $(date)"
echo "SLURM_JOB_ID:    ${SLURM_JOB_ID}"

PROJECT_NAME=pytorch-cifar	# if you're changing this, remember to change the destination of the logs (header of this file) as well
USER=pbelcak				# if you're changing this, remember to change the user of the logs (header of this file) as well
PROJECT_PATH=/home/${USER}/${PROJECT_NAME}
STORAGE_PATH=/itet-stor/${USER}/net_scratch/${PROJECT_NAME}

ARCHITECTURE=${1:-vgg19_resilu_block}
CHECKPOINT=$2

# Binary or script to execute
PYTHONPATH=${PROJECT_PATH} python ${PROJECT_PATH}/main.py \
	--job_id=${SLURM_JOB_ID} \
	--job_suite=${ARCHITECTURE} \
	--seed=0 \
	--data_directory=${STORAGE_PATH}/data/ \
	--checkpointing_directory=${STORAGE_PATH}/checkpoints/ \
	--logging_directory=${STORAGE_PATH}/logs/ \
	--results_directory=${STORAGE_PATH}/results/ \
	--image_directory=${STORAGE_PATH}/images/ \
	--action=build_image \
	--dataset=cifar10 \
	--architecture=${ARCHITECTURE} \
	--checkpoint=${CHECKPOINT} \
	--batch_size=128 \
	--evaluate_after_training ${@:3}

# Send more noteworthy information to the output log
echo "Finished at:     $(date)"

# End the script with exit code 0
exit 0