#!/bin/bash
#SBATCH --job-name=mae_pretrain
#SBATCH --output=/beegfs/work/mae_entr/mae/slurm/out/%x_%j.out # Standard output log (%x = job name, %j = job ID)
#SBATCH --error=/beegfs/work/mae_entr/mae/slurm/err/%x_%j.err  # Standard error log
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --gres=gpu:1080:2
#SBATCH --partition=gpu
#SBATCH --nodelist=gpu03
#SBATCH --time=1-00:00:00

# Create the output directories if they don't exist
mkdir -p /beegfs/work/mae_entr/mae/slurm/out /beegfs/work/mae_entr/mae/slurm/err

module load anaconda/3-5.0.1
module load cuda/12.2
source activate /beegfs/work/mae_entr/conda_envs/mae_env
echo "Activated conda environment successfully."


# Generate a timestamp
timestamp=$(date +"%Y%m%d%H%M%S")

infostr="_Pretrain_IMNET1K_epoch20_entropyreverse_entropyweighted_ratio50_warmup2_modelbase"

# Create new directory with timestamp under ./jobs
newDir="/beegfs/work/mae_entr/mae/jobs/${timestamp}${infostr}"
# newDir="./jobs/test"
mkdir -p "$newDir"

# Define masking type and arguments
masking_type="random_masking"
masking_args='{"masking_ratio":0.75}'
# masking_args=$(echo '{"masking_ratio": 0.75}' | jq -c . | sed 's/"/\\"/g')

# Define configuration variables
dataPath="/beegfs/data/shared/imagenet/imagenet100/train/"
outputDir="$newDir/outputs"
logDir="$newDir/logs"
batchSize=48
epochs=1
accumIter=$((4096 / batchSize))
model="mae_vit_large_patch16"
inputSize=224
weightDecay=0.05
blr=1.5e-4
minLr=0
warmupEpochs=2
device="cuda"
seed=0
resume=""
startEpoch=0
numWorkers=8
persistentWorkers=true
pinMem=true
worldSize=1
localRank=-1
distUrl="env://"
normPixLoss=false
checkpoint_freq=2

# Construct the training command with mandatory arguments
trainingCommand="python ./main_pretrain.py --data_path $dataPath"
trainingCommand+=" --output_dir $outputDir"
trainingCommand+=" --log_dir $logDir"
trainingCommand+=" --batch_size $batchSize"
trainingCommand+=" --epochs $epochs"
trainingCommand+=" --accum_iter $accumIter"
trainingCommand+=" --model $model"
trainingCommand+=" --input_size $inputSize"
trainingCommand+=" --masking_type $masking_type"
trainingCommand+=" --masking_args '$masking_args'" # Note: masking_args is a JSON string, single quotes are necessary
trainingCommand+=" --weight_decay $weightDecay"
trainingCommand+=" --blr $blr"
trainingCommand+=" --min_lr $minLr"
trainingCommand+=" --warmup_epochs $warmupEpochs"
trainingCommand+=" --device $device"
trainingCommand+=" --seed $seed"
trainingCommand+=" --start_epoch $startEpoch"
trainingCommand+=" --num_workers $numWorkers"
trainingCommand+=" --world_size $worldSize"
trainingCommand+=" --local_rank $localRank"
trainingCommand+=" --dist_url $distUrl"
trainingCommand+=" --checkpoint_freq $checkpoint_freq"

# Add conditional arguments
if [ -n "$resume" ]; then
    trainingCommand+=" --resume \"$resume\""
fi
if [ "$normPixLoss" = true ]; then
    trainingCommand+=" --norm_pix_loss"
fi
if [ "$pinMem" = true ]; then
    trainingCommand+=" --pin_mem"
fi
if [ "$persistentWorkers" = true ]; then
    trainingCommand+=" --persistent_workers"
fi

# Echo the training command
echo "Training Command: $trainingCommand"

# Execute the training command
srun $trainingCommand

# Optional: Add any post-training commands here, like logging or sending a notification
