$python_cmd = "uv run python"
# Full run configuration: N=120, 10 epochs, 50 steps per epoch
$common_args = "120 --ants 20 --val_ants 50 --batch_size 2 --steps 50 --epochs 10 --val_size 10 --disable_wandb"

New-Item -ItemType Directory -Force -Path "pretrained/bpp"
Push-Location bpp

# Clean up previous logs
Remove-Item "../final_results_*.log" -ErrorAction SilentlyContinue
Remove-Item "../final_results.log" -ErrorAction SilentlyContinue

Function Run-Experiment ($name, $exp_args, $log_suffix) {
    $log_file = "../final_results_${log_suffix}.log"
    Write-Host "Running $name..."
    Add-Content $log_file "=================================================="
    Add-Content $log_file "Running $name..."
    Add-Content $log_file "=================================================="
    # Use Out-File with encoding and Append
    Invoke-Expression "$python_cmd train.py $exp_args" | Out-File -FilePath $log_file -Append -Encoding utf8
    Get-Content $log_file -Tail 5
}

# 2. +Arch
Run-Experiment "2/6: +Architecture (GatedGCN) (N=120)" "$common_args --run_name Arch --model_type gcn --loss_type mse" "2"

# 3. +Loss
Run-Experiment "3/6: +Loss (Huber) (N=120)" "$common_args --run_name Loss --model_type embnet --loss_type huber" "3"

# 4. +MMAS
Run-Experiment "4/6: +MMAS (N=120)" "$common_args --run_name MMAS --model_type embnet --loss_type mse --use_mmas" "4"

# 5. +LS
Run-Experiment "5/6: +Local Search (N=120)" "$common_args --run_name LS --model_type embnet --loss_type mse --use_local_search" "5"

# 6. Combined
Run-Experiment "6/6: Combined Improvements (N=120)" "$common_args --run_name Combined --model_type gcn --loss_type huber --use_mmas --use_local_search" "6"

Pop-Location

# Combine logs for analysis
Get-Content final_results_*.log | Out-File -FilePath final_results.log -Encoding utf8
Write-Host "Final experiments completed. Combined log saved to final_results.log"
