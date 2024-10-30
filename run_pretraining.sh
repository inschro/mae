#!/bin/bash

# Generate a timestamp
timestamp=$(date +"%Y%m%d%H%M%S")

infostr="_Pretrain_IMNET1K_epoch20_entropyreverse_entropyweighted_ratio50_warmup2_modelbase"

# Create new directory with timestamp under ./jobs
newDir="./jobs/${timestamp}${infostr}"
# newDir="./jobs/test"
mkdir -p "$newDir"

# Define masking type and arguments
masking_type="entropy_masking"
masking_args=$(echo '{"masking_ratio": 0.50, "reverse": true}' | jq -c . | sed 's/"/\\"/g')

# Define configuration variables
dataPath="/home/darius/Dokumente/Research/mae/data/imagnet-mini/train"
outputDir="$newDir/outputs"
logDir="$newDir/logs"
batchSize=48
epochs=20
accumIter=$((4096 / batchSize))
model="mae_vit_base_patch16"
inputSize=224
weightDecay=0.05
blr=1.5e-4
minLr=0
warmupEpochs=2
device="cuda"
seed=1
resume=""
startEpoch=0
numWorkers=6
persistentWorkers=true
pinMem=true
worldSize=1
localRank=-1
distUrl="env://"
normPixLoss=false
checkpoint_freq=2

# Activate Python environment (adjust path as necessary)
conda activate ml

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
                --dist_url $distUrl \
                --checkpoint_freq $checkpoint_freq"

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
eval "$trainingCommand"

# Optional: Add any post-training commands here, like logging or sending a notification
