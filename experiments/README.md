# GFACS BPP Experiments

This directory contains the configurations and scripts used to test domain-specific enhancements for the Bin Packing Problem (BPP) using the Generative Flow Ant Colony Sampler (GFACS) framework.

## Directory Structure

*   `bpp/`: Experimental fork of the original baseline code.
*   `data/`: Copied datasets required for BPP evaluation.
*   `scripts/`: Python utility scripts (e.g., `analyze_results.py` and data generators).
*   `artifacts/`: Generated plots and csv files for results analysis.
*   `results_archive/`: Historical raw `.log` outputs from previous test runs.
*   `experiments.md`: A detailed report on the methodology, heuristics designed, and final results.

## Execution Requirements
Ensure that the `data/` folder from the main GFACS repository is copied into this `experiments/` directory before running any scripts. (See the main repository `README.md` for `cp` commands). Furthermore, ensure you have `uv` installed for dependency management.

## How to Run Experiments

Both Bash (`.sh`) and PowerShell (`.ps1`) scripts are provided depending on your environment. The scripts automatically iterate through permutations of the BPP enhancements.

**1. Quick Test Run (1 Epoch)**
This will quickly test all 6 configurations for exactly 1 epoch to ensure integration works properly.
```bash
./run_experiments.sh
# OR
./run_experiments.ps1
```

**2. Rigorous Run (10 Epochs)**
This replicates the full length of the experiments used for the data reported in `experiments.md`.
```bash
./run_final_experiments.sh
# OR
./run_final_experiments.ps1
```

**3. Test Environment Check**
A smaller scaled script for simple integration checking.
```bash
./run_test_experiments.sh
# OR
./run_test_experiments.ps1
```

## Analyzing Results
After running the scripts, a `.log` file will be generated in the root of the `experiments/` folder (e.g., `final_results.log`).

You can parse this log and automatically regenerate the `results_fitness.png` and `results_bins.png` images by running the analysis script:
```bash
uv run python scripts/analyze_results.py
```
