# Generate a timestamp
$timestamp = Get-Date -Format "yyyyMMddHHmmss"

# Create new directory with timestamp under .\jobs
$newDir = ".\jobs\$timestamp finetune"
New-Item -ItemType Directory -Force -Path $newDir

# Define configuration variables
$batchSize = 64
$epochs = 100
$accumIter = 16

# Model Parameters
$model = "vit_base_patch16"
$inputSize = 224
$drop_path = 0.1

# Optimizer Parameters
$clip_grad = 0.2
$weightDecay = 0.0
$blr = 1e-3
$lr = $null
$layer_decay = 0.75
$min_lr = 1e-6
$warmupEpochs = 5

# Augmentation Parameters
$color_jitter = $null
$aa = "rand-m9-mstd0.5-inc1"
$smoothing = 0.1

# Random erase parameters
$reprob = 0.25
$remode = "pixel"
$recount = 1
$resplit = $false

# Mixup parameters
$mixup = 0.8
$cutmix = 1.0
$mixup_prob = 1.0
$mixup_switch_prob = 0.5
$mixup_mode = "batch"

# Finetuning parameters
$finetune = "C:\Users\Ingo\Desktop\Code Stuff\mae\mae\jobs\20240821214220\outputs\checkpoint-799.pth"

# Dataset parameters
$dataPath = "C:\Users\Ingo\Desktop\imagenet-mini"
$nbClasses = 1000
$outputDir = "$newDir\outputs"
$logDir = "$newDir\logs"
$device = "cuda"
$seed = 0
$resume = ""
$startEpoch = 0
$eval = $false
$numWorkers = 10
$pinMem = $true

# other parameters
$print_freq = 100


# Construct the argument string
$arguments = @(
    "--batch_size $batchSize",
    "--epochs $epochs",
    "--accum_iter $accumIter",
    "--model $model",
    "--input_size $inputSize",
    "--drop_path $drop_path",
    "--weight_decay $weightDecay",
    "--blr $blr",
    "--layer_decay $layer_decay",
    "--min_lr $min_lr",
    "--warmup_epochs $warmupEpochs",
    "--aa $aa",
    "--smoothing $smoothing",
    "--reprob $reprob",
    "--remode $remode",
    "--recount $recount",
    "--resplit",
    "--mixup $mixup",
    "--cutmix $cutmix",
    "--mixup_prob $mixup_prob",
    "--mixup_switch_prob $mixup_switch_prob",
    "--mixup_mode $mixup_mode",
    "--finetune `"$finetune`"",
    "--cls_token",
    "--data_path `"$dataPath`"",
    "--nb_classes $nbClasses",
    "--output_dir `"$outputDir`"",
    "--log_dir `"$logDir`"",
    "--device $device",
    "--seed $seed",
    "--start_epoch $startEpoch",
    "--num_workers $numWorkers",
    "--print_freq $print_freq"
)

if ($pinMem) {
    $arguments += "--pin_mem"
} else {
    $arguments += "--no_pin_mem"
}
if ($eval) {
    $arguments += "--eval"
}
if ($resplit) {
    $arguments += "--resplit"
}
if ($null -ne $clip_grad) {
    $arguments += "--clip_grad $clip_grad"
}
if ($null -ne $lr) {
    $arguments += "--lr $lr"
}
if ($null -ne $color_jitter) {
    $arguments += "--color_jitter $color_jitter"
}
if ($resume -ne "") {
    $arguments += "--resume `"$resume`""
}

# Join the arguments into a single string
$argsString = $arguments -join ' '

# Run the Python script with the arguments and print the command
$trainingCommand = "python main_finetune.py $argsString"
Write-Host $trainingCommand
Invoke-Expression $trainingCommand