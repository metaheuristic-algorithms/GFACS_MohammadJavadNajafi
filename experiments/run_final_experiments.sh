#!/bin/bash
python_cmd="uv run python"
common_args="120 --ants 20 --val_ants 50 --batch_size 2 --steps 50 --epochs 10 --val_size 10 --disable_wandb"

mkdir -p pretrained/bpp
cd bpp || exit

rm -f ../final_results_*.log
rm -f ../final_results.log

run_experiment() {
    local name="$1"
    local exp_args="$2"
    local log_suffix="$3"
    local log_file="../final_results_${log_suffix}.log"
    
    echo "Running $name..."
    echo "==================================================" >> "$log_file"
    echo "Running $name..." >> "$log_file"
    echo "==================================================" >> "$log_file"
    $python_cmd train.py $exp_args >> "$log_file" 2>&1
    tail -n 5 "$log_file"
}

run_experiment "2/6: +Architecture (GatedGCN) (N=120)" "$common_args --run_name Arch --model_type gcn --loss_type mse" "2"
run_experiment "3/6: +Loss (Huber) (N=120)" "$common_args --run_name Loss --model_type embnet --loss_type huber" "3"
run_experiment "4/6: +MMAS (N=120)" "$common_args --run_name MMAS --model_type embnet --loss_type mse --use_mmas" "4"
run_experiment "5/6: +Local Search (N=120)" "$common_args --run_name LS --model_type embnet --loss_type mse --use_local_search" "5"
run_experiment "6/6: Combined Improvements (N=120)" "$common_args --run_name Combined --model_type gcn --loss_type huber --use_mmas --use_local_search" "6"

cd ..
cat final_results_*.log > final_results.log
echo "Final experiments completed. Combined log saved to final_results.log"
