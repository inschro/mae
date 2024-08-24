# Generate a timestamp
$timestamp = Get-Date -Format "yyyyMMddHHmmss"

# Create new directory with timestamp under .\jobs
$newDir = ".\jobs\$timestamp"
$newDir = ".\jobs\20240821214220"
New-Item -ItemType Directory -Force -Path $newDir

# Define masking type and arguments
$masking_type = "entropy_masking_bins"
$masking_args = @{
    ratios = @(0.8, 0.8, 1, 1)
    # masking_ratio = 0.95
} | ConvertTo-Json -Compress

$masking_args = $masking_args -replace '"', '\"'

# Define configuration variables
$dataPath = "C:\Users\Ingo\Desktop\imagenet-mini\train"
$outputDir = "$newDir\outputs"
$logDir = "$newDir\logs"
$batchSize = 32
$epochs = 800
$accumIter = 32
$model = "mae_vit_base_patch16"
$inputSize = 224
$lr = $null
$weightDecay = 0.05
$blr = 1.5e-4
$minLr = 0
$warmupEpochs = 40
$device = "cuda"
$seed = 0
$resume = "C:\Users\Ingo\Desktop\Code Stuff\mae\mae\jobs\20240821214220\outputs\checkpoint-700.pth"
$startEpoch = 701
$numWorkers = 10
$pinMem = $true
$persistentWorkers = $true
$worldSize = 1
$localRank = -1
$distUrl = "env://"
$normPixLoss = $false

# Activate Python environment (adjust path as necessary)
& .\.mae-env\Scripts\activate

# Construct the training command with mandatory arguments
$trainingCommand = "python .\main_pretrain.py --data_path $dataPath " +
                    "--output_dir $outputDir " +
                    "--log_dir $logDir " +
                    "--batch_size $batchSize " +
                    "--epochs $epochs " +
                    "--accum_iter $accumIter " +
                    "--model $model " +
                    "--input_size $inputSize " +
                    "--masking_type $masking_type " +
                    "--masking_args `'$masking_args`' " +
                    "--weight_decay $weightDecay " +
                    "--blr $blr " +
                    "--min_lr $minLr " +
                    "--warmup_epochs $warmupEpochs " +
                    "--device $device " +
                    "--seed $seed " +
                    "--start_epoch $startEpoch " +
                    "--num_workers $numWorkers " +
                    "--world_size $worldSize " +
                    "--local_rank $localRank " +
                    "--dist_url $distUrl"

# Add conditional arguments
if ($null -ne $lr) {
    $trainingCommand += " --lr $lr"
}
if ($resume -ne "") {
    $trainingCommand += " --resume `"$resume`""
}
if ($normPixLoss) {
    $trainingCommand += " --norm_pix_loss"
}
if ($pinMem) {
    $trainingCommand += " --pin_mem"
} else {
    $trainingCommand += " --no_pin_mem"
}
if ($persistentWorkers) {
    $trainingCommand += " --persistent_workers"
}

# Echo the training command
Write-Output "Training Command: $trainingCommand"

# Execute the training command
Invoke-Expression $trainingCommand

# Optional: Add any post-training commands here, like logging or sending a notification