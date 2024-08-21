# Define the arguments
$batch_size = 4
$epochs = 50
$accum_iter = 1
$model = 'vit_base_patch16'
$input_size = 224
$weight_decay = 0.05
$lr = $null
$blr = 1e-3
$layer_decay = 0.75
$min_lr = 1e-6
$warmup_epochs = 5


$finetune = 'C:\Users\Ingo\Desktop\Code Stuff\mae\mae\mae_pretrain_vit_base.pth'
$global_pool = $true
$data_path = 'C:\Users\Ingo\Desktop\imagenet-mini'
$nb_classes = 1000
$output_dir = 'C:\Users\Ingo\Desktop\Code Stuff\mae\mae\jobs\downstream\classification\output_dir'
$log_dir = 'C:\Users\Ingo\Desktop\Code Stuff\mae\mae\jobs\downstream\classification\log_dir'
$device = 'cuda'
$seed = 0
$resume = 'C:\Users\Ingo\Desktop\Code Stuff\mae\mae\mae_pretrain_vit_base.pth'
$start_epoch = 0
$eval = $false
$dist_eval = $false
$num_workers = 10
$pin_mem = $true
$no_pin_mem = $false
$world_size = 1
$local_rank = -1
$dist_on_itp = $false
$dist_url = 'env://'

# Construct the argument string
$arguments = @(
    "--batch_size $batch_size",
    "--epochs $epochs",
    "--accum_iter $accum_iter",
    "--model $model",
    "--input_size $input_size",
    "--weight_decay $weight_decay",
    "--lr $lr",
    "--blr $blr",
    "--layer_decay $layer_decay",
    "--min_lr $min_lr",
    "--warmup_epochs $warmup_epochs",
    "--finetune $finetune",
    "--global_pool $global_pool",
    "--data_path $data_path",
    "--nb_classes $nb_classes",
    "--output_dir $output_dir",
    "--log_dir $log_dir",
    "--device $device",
    "--seed $seed",
    "--resume $resume",
    "--start_epoch $start_epoch",
    "--eval $eval",
    "--dist_eval $dist_eval",
    "--num_workers $num_workers",
    "--pin_mem $pin_mem",
    "--no_pin_mem $no_pin_mem",
    "--world_size $world_size",
    "--local_rank $local_rank",
    "--dist_on_itp $dist_on_itp",
    "--dist_url $dist_url"
)

# Remove null or false arguments
$arguments = $arguments | Where-Object { 
    $_ -notmatch '--clip_grad $null' -and 
    $_ -notmatch '--lr $null' -and 
    $_ -notmatch '--color_jitter $null' -and 
    $_ -notmatch '--cutmix_minmax $null' -and 
    $_ -notmatch '--resplit $false' -and 
    $_ -notmatch '--eval $false' -and 
    $_ -notmatch '--dist_eval $false' -and 
    $_ -notmatch '--no_pin_mem $false' -and 
    $_ -notmatch '--dist_on_itp $false' -and 
    $_ -notmatch '--global_pool $false'
}

# Join the arguments into a single string
$argsString = $arguments -join ' '

# Run the Python script with the arguments
python main_finetune.py $argsString