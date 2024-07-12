#!/bin/bash

# Generate a timestamp
timestamp=$(date +"%Y%m%d%H%M%S")

# Create new directory with timestamp under ./jobs
newDir="./jobs/$timestamp"
mkdir -p "$newDir"

# Define masking type and arguments
masking_type="entropy_masking"
masking_args=$(echo '{"masking_ratio": 0.7}' | jq -c . | sed 's/"/\\"/g')

# Define configuration variables
dataPath="./data/imagnet-mini/val"
outputDir="$newDir/outputs"
logDir="$newDir/logs"
batchSize=30
epochs=100
accumIter=4
model="mae_vit_base_patch16"
inputSize=224
lr=1e-4
weightDecay=0.05
blr=1e-3
minLr=3e-5
warmupEpochs=1
device="cuda"
seed=0
resume=""
startEpoch=0
numWorkers=10
pinMem=true
worldSize=1
localRank=-1
distUrl="env://"
normPixLoss=false

# Activate Python environment (adjust path as necessary)
conda activate mae

# Construct the training command with mandatory arguments
trainingCommand="python ./main_pretrain.py --data_path $dataPath \
                   --output_dir $outputDir \
                   --log_dir $logDir \
                   --batch_size $batchSize \
                   --epochs $epochs \
                   --accum_iter $accumIter \
                   --model $model \
                   --input_size $inputSize \
                   --masking_type $masking_type \
                   --masking_args \"$masking_args\" \
                   --lr $lr \
                   --weight_decay $weightDecay \
                   --blr $blr \
                   --min_lr $minLr \
                   --warmup_epochs $warmupEpochs \
                   --device $device \
                   --seed $seed \
                   --start_epoch $startEpoch \
                   --num_workers $numWorkers \
                   --world_size $worldSize \
                   --local_rank $localRank \
                   --dist_url $distUrl"

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

# Echo the training command
echo "Training Command: $trainingCommand"

# Execute the training command
eval "$trainingCommand"

# Optional: Add any post-training commands here, like logging or sending a notification
