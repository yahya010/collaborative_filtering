# Collaborative Filtering Methods for Paper Recommendation Systems

## Overview
This project implements a modular pipeline for building, evaluating, and ensembling recommendation models using PyTorch and geometric approaches. The core logic is contained in the provided Jupyter notebook (`final_combined.ipynb`).

##Colab Link: https://colab.research.google.com/drive/15noB-aEWY_43C2eUd5SBAtembEjoIw78?usp=sharing

## Prerequisites
- Python 3.7 or higher
- A virtual environment tool (e.g., `uv`, `venv`, or `conda`)

## Setup
1. **Clone the repository**:
   ```bash
   git clone <repository_url>
   cd <repository_directory>
   ```
2. **Create and activate a virtual environment** (using `uv`):
   ```bash
   uv create env
   uv activate env      # on Linux/MacOS and Windows
   ```
   *(Alternatively, use `venv` or `conda` if preferred.)*
3. **Install dependencies**:
   ```bash
   pip install pandas numpy matplotlib torch torch-geometric torch-sparse scikit-learn
   ```

## Running the Notebook
1. Launch Jupyter Notebook:
   ```bash
   jupyter notebook
   ```
2. Open `final_combined.ipynb` and run all cells. The code should execute successfully once all dependencies are installed.

## Implemented Methods
In this notebook, we apply the following methods:

1. Optimal Rank-\(k\) SVD
2. Embedding Dot-Product Model
3. Iterative SVD
4. SVD++
5. NeuMF
6. GraphNeuMF
7. DMF
8. Ensemble strategies (simple average, weighted average, top-\(k\) weighted, deep stacking)

## Notes
- Ensure you have sufficient memory for data loading and model training.
- For GPU acceleration, verify that your PyTorch installation includes CUDA support.
