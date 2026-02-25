$python_cmd = "uv run python"
# Configuration for N=120 (Standard size, 10 epochs for speed)
# N=120 beta values are in the map in train.py, so we don't strictly need to pass them, 
# but passing them ensures consistency if we want to tweak.
# Default map: {120: 500, ...} for min and {120: 5000, ...} for max.
# We will use 10 epochs to fit in ~2 hours total (6 runs).
$common_args = "120 --ants 20 --val_ants 50 --batch_size 2 --steps 20 --epochs 1 --val_size 10 --disable_wandb --val_interval 1"

# Create output directory
New-Item -ItemType Directory -Force -Path "pretrained/bpp"

Push-Location bpp

$log_file = "../experiment_results.log"
Clear-Content $log_file -ErrorAction SilentlyContinue

Function Run-Experiment ($name, $exp_args) {
    Write-Host "Running $name..."
    Add-Content $log_file "=================================================="
    Add-Content $log_file "Running $name..."
    Add-Content $log_file "=================================================="
    # Use Invoke-Expression and pipe output to file and host
    Invoke-Expression "$python_cmd train.py $exp_args" | Tee-Object -FilePath $log_file -Append
}

# 1. Base (Original EmbNet, MSE Loss, No Heuristics)
Run-Experiment "1/6: Base Model (N=120)" "$common_args --run_name Base --model_type embnet --loss_type mse"

# 2. +Arch (GatedGCN, MSE, No Heuristics)
Run-Experiment "2/6: +Architecture (GatedGCN) (N=120)" "$common_args --run_name Arch --model_type gcn --loss_type mse"

# 3. +Loss (Original EmbNet, Huber Loss, No Heuristics)
Run-Experiment "3/6: +Loss (Huber) (N=120)" "$common_args --run_name Loss --model_type embnet --loss_type huber"

# 4. +MMAS (Original, MSE, MMAS)
Run-Experiment "4/6: +MMAS (N=120)" "$common_args --run_name MMAS --model_type embnet --loss_type mse --use_mmas"

# 5. +LS (Original, MSE, Local Search)
Run-Experiment "5/6: +Local Search (N=120)" "$common_args --run_name LS --model_type embnet --loss_type mse --use_local_search"

# 6. Combined (GatedGCN, Huber, MMAS, Local Search)
Run-Experiment "6/6: Combined Improvements (N=120)" "$common_args --run_name Combined --model_type gcn --loss_type huber --use_mmas --use_local_search"

Pop-Location

Write-Host "All experiments completed. Results saved to $log_file"
