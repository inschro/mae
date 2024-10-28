#!/bin/bash

# Generate a timestamp
timestamp=$(date +"%Y%m%d%H%M%S")

# Create new directory with timestamp under ./jobs
newDir="./jobs/${timestamp}_linprobe_entropyreverse_ratio75_warmup2_modelbase"
mkdir -p "$newDir"

# Define configuration variables
batchSize=256
epochs=90
accumIter=32
model="vit_base_patch16"
weightDecay=0.0
blr=0.1
warmupEpochs=10
finetune="/home/ingo/Desktop/code_stuff/mae/jobs/20241001182847_Pretrain_IMNET1K_epoch20_entropyreverse_ratio75_warmup2_modelbase/outputs/checkpoint-19.pth"
dataPath="/media/ingo/539ea23b-a9e6-475b-993c-4f8f7eab2ac0/imagenet-mini/"
nbClasses=1000
outputDir="$newDir/outputs"
logDir="$newDir/logs"
device="cuda"
seed=0
startEpoch=0
eval=false
numWorkers=10
pinMem=true
distUrl="env://"

# Construct the argument string
arguments=(
    "--batch_size $batchSize"
    "--epochs $epochs"
    "--accum_iter $accumIter"
    "--model $model"
    "--cls_token"
    "--weight_decay $weightDecay"
    "--blr $blr"
    "--warmup_epochs $warmupEpochs"
    "--data_path \"$dataPath\""
    "--nb_classes $nbClasses"
    "--output_dir \"$outputDir\""
    "--log_dir \"$logDir\""
    "--device $device"
    "--seed $seed"
    "--start_epoch $startEpoch"
    "--num_workers $numWorkers"
)

if [ -n "$finetune" ]; then
    arguments+=("--finetune \"$finetune\"")
fi

if [ "$pinMem" = true ]; then
    arguments+=("--pin_mem")
else
    arguments+=("--no_pin_mem")
fi

if [ -n "$distUrl" ]; then
    arguments+=("--dist_url $distUrl")
fi

if [ "$eval" = true ]; then
    arguments+=("--eval")
fi

# Join the arguments into a single string
argsString=$(printf " %s" "${arguments[@]}")

# Run the Python script with the arguments and print the command
trainingCommand="python main_linprobe.py$argsString"
echo $trainingCommand
eval $trainingCommand