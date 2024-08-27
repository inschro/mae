# Generate a timestamp
$timestamp = Get-Date -Format "yyyyMMddHHmmss"

# Create new directory with timestamp under .\jobs
$newDir = ".\jobs\$timestamp linprobe"
New-Item -ItemType Directory -Force -Path $newDir

# Define configuration variables
$batchSize = 256
$epochs = 90
$accumIter = 2
$model = "vit_base_patch16"
$weightDecay = 0.0
$blr = 0.1
$warmupEpochs = 10
$finetune = "C:\Users\Ingo\Desktop\Code Stuff\mae\mae\jobs\20240821214220\outputs\checkpoint-799.pth"
$dataPath = "C:\Users\Ingo\Desktop\imagenet-mini"
$nbClasses = 1000
$outputDir = "$newDir\outputs"
$logDir = "$newDir\logs"
$device = "cuda"
$seed = 0
$startEpoch = 0
$eval = $false
$numWorkers = 10
$pinMem = $true
$distUrl = "env://"

# Construct the argument string
$arguments = @(
    "--batch_size $batchSize",
    "--epochs $epochs",
    "--accum_iter $accumIter",
    "--model $model",
    "--cls_token",
    "--weight_decay $weightDecay",
    "--blr $blr",
    "--warmup_epochs $warmupEpochs",
    "--data_path `"$dataPath`"",
    "--nb_classes $nbClasses",
    "--output_dir `"$outputDir`"",
    "--log_dir `"$logDir`"",
    "--device $device",
    "--seed $seed",
    "--start_epoch $startEpoch",
    "--num_workers $numWorkers"
)

if ($finetune -ne "") {
    $arguments += "--finetune `"$finetune`""
}

if ($pinMem) {
    $arguments += "--pin_mem"
} else {
    $arguments += "--no_pin_mem"
}

if ($distUrl -ne "") {
    $arguments += "--dist_url $distUrl"
}

if ($eval) {
    $arguments += "--eval"
}

# Join the arguments into a single string
$argsString = $arguments -join ' '


# Run the Python script with the arguments and print the command
$trainingCommand = "python main_linprobe.py $argsString"
Write-Host $trainingCommand
Invoke-Expression $trainingCommand