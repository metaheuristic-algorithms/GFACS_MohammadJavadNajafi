# GFACS: Generative Flow Ant Colony Sampler

**Name:** Mohammad Javad Najafi  
**Student ID:** 403131026  
**Course:** Metaheuristics course, Amirkabir University of Technology

This repository contains the code, experiments, and associated literature for the Generative Flow Ant Colony Sampler (GFACS) project.

## Project Structure

The repository has been organized into four primary directories:

### 1. `original_code/`
Contains the original, unmodified source code for the GFACS algorithm as provided by the original authors (Kim et al., 2025). This includes the baseline models and training scripts for various Combinatorial Optimization problems like the Traveling Salesman Problem (TSP), Capacitated Vehicle Routing Problem (CVRP), and the Bin Packing Problem (BPP).

*See `original_code/README.md` for specific execution instructions on the baselines.*

### 2. `experiments/`
Contains additional experiments and enhancements built on top of the original framework, specifically focusing on the **Bin Packing Problem (BPP)**.
- **Included Additions**: Huber Loss stabilization, Max-Min Ant System (MMAS) pheromone clamping, Swap/2-opt Local Search refinement, and GatedGCN architecture tests.
- **Independence**: This folder includes its own copy of the necessary source files (`bpp/`, `data/`) so experiments can be run independently using the provided `run_experiments.sh` and `run_final_experiments.sh` scripts without modifying the original code.
- **Results**: See `experiments/experiments.md` for a detailed breakdown of the methodologies, execution times, and performance improvements over the baseline.

### 3. `report/`
Contains the LaTeX source code and compiled PDF for the revised academic paper.
- The `edited.tex` file integrates the findings from the `experiments/` folder into the original paper content, clearly highlighting our contributions and extensions to the BPP.
- Includes updated tables, figures, and methodology formulas.

### 4. `presentation/`
Contains presentation materials summarizing the project's methodologies and empirical successes. 
- **Video Link:** The presentation video can be accessed directly here: [GFACS Presentation Video](https://drive.google.com/file/d/1esADKYuhcOMfC9nkeJg-K-Kw3E_Q5RZF/view?usp=sharing). 
- Note that this link is also provided in `presentation/gdrive-video-link.txt` and `presentation/README.md`.

## Getting Started

### 1. Cloning the Repository (with Video)
Because the presentation video is large, it is tracked using Git Large File Storage (LFS). To clone this repository and correctly pull the video file, ensure you have [Git LFS](https://git-lfs.com/) installed and run:

```bash
git clone https://github.com/metaheuristic-algorithms/GFACS_MohammadJavadNajafi.git
cd GFACS_MohammadJavadNajafi
git lfs pull
```

### 2. Requirements and Data Initialization
Due to size constraints, the raw dataset instances are not included directly in this repository. To run any experiments, you must first clone the original GFACS repository and copy its `data/` folder:

```bash
# Clone the original repository using uv
git clone https://github.com/ai4co/gfacs.git original_gfacs
# Copy the data folder to both execution locations
cp -r original_gfacs/data original_code/
cp -r original_gfacs/data experiments/
```

Make sure you have `uv` installed to handle the python environments seamlessly.

### 3. Running the Original Baselines
To explore the original baselines, navigate to `original_code/`. See `original_code/README.md` for specific command configurations.
```bash
cd original_code/
uv run python bpp/train.py <arguments>
```

### 4. Running the BPP Extensions
To replicate our specific BPP enhancements, navigate to `experiments/`. We have provided convenient bash execution scripts that reproduce the 6 BPP configurations automatically.

```bash
cd experiments/
# Make sure the scripts are executable
chmod +x run_experiments.sh
chmod +x run_final_experiments.sh

# Run the 1-epoch quick experiments
./run_experiments.sh

# Run the full 10-epoch rigorous experiments
./run_final_experiments.sh
```

Logs will be generated sequentially (e.g., `experiment_results_2.log`) and then combined into a master log (e.g., `experiment_results.log`). Data analysis figures can then be regenerated using `uv run python analyze_results.py`.
