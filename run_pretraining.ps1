# Generate a timestamp
$timestamp = Get-Date -Format "yyyyMMddHHmmss"

# Create new directory with timestamp under .\jobs
$newDir = ".\jobs\$timestamp"
New-Item -ItemType Directory -Force -Path $newDir

# Define configuration variables
$dataPath = "C:\Users\Ingo\Desktop\imagenet-mini\val"
$outputDir = "$newDir\outputs"
$logDir = "$newDir\logs"
$batchSize = 4
$epochs = 50
$accumIter = 4
$model = "mae_vit_large_patch16"
$inputSize = 224
$maskRatio = 0.9
$masking = "entropy" # "random" or "entropy"
$lr = 1e-4
$weightDecay = 0.05
$blr = 1e-3
$minLr = 3e-5
$warmupEpochs = 1
$device = "cuda"
$seed = 0
$resume = ".\mae_visualize_vit_large.pth"
$startEpoch = 1
$numWorkers = 10
$pinMem = $true
$worldSize = 1
$localRank = -1
$distUrl = "env://"
$normPixLoss = $false

# Activate Python environment (adjust path as necessary)
& .\.mae-env\Scripts\activate

# Construct the training command with mandatory arguments
$trainingCommand = "python .\main_pretrain.py --data_path $dataPath --output_dir $outputDir --log_dir $logDir --batch_size $batchSize --epochs $epochs --accum_iter $accumIter --model $model --input_size $inputSize --mask_ratio $maskRatio --masking $masking --lr $lr --weight_decay $weightDecay --blr $blr --min_lr $minLr --warmup_epochs $warmupEpochs --device $device --seed $seed --start_epoch $startEpoch --num_workers $numWorkers --world_size $worldSize --local_rank $localRank --dist_url $distUrl"

# Add conditional arguments
if ($resume -ne "") {
    $trainingCommand += " --resume `"$resume`""
}
if ($normPixLoss) {
    $trainingCommand += " --norm_pix_loss"
}
if ($pinMem) {
    $trainingCommand += " --pin_mem"
}

# Execute the training command
Invoke-Expression $trainingCommand

# Optional: Add any post-training commands here, like logging or sending a notification