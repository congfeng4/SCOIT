# HGNN: Hypergraph Neural Network for Graph Representation Learning

A hypergraph neural network framework based on hypergraphs for learning node representations and performing graph-based tasks. This repository provides implementations of hypergraph neural networks with a focus on adjacency matrix (adj) features for graph learning tasks.

## Overview

HGNN (Hypergraph Neural Network) is designed to handle hypergraph structures where edges (hyperedges) can connect multiple nodes simultaneously. This framework extends traditional graph neural networks by capturing complex relationships between nodes in complex systems, with a specific emphasis on using adjacency matrix features.

Key features:
- Hypergraph representation and processing
- Focus on adjacency matrix (adj) features for learning
- Support for various loss functions (BCE, MSE, ZINB, etc.)
- GPU acceleration with PyTorch

## Installation

### Prerequisites

- numpy
- scipy 1.6.0
- sklearn
- communities
- igraph
- leidenalg
- pytorch 1.9.0
- Additional dependencies listed in `requirements.txt`

### Install from Source

```bash
# Clone the repository
git clone https://github.com/your-username/HGNN.git
cd HGNN

# Create and activate conda environment (optional but recommended)
conda env create -f env-py39.yaml
conda activate py39

# Install package
pip install .
```

### Alternative Installation

```bash
pip install -r requirements.txt
```

## Getting Started

### Example Notebooks

The repository provides Jupyter Notebook examples for quick start and detailed usage:

- `hgnn_sc_GEM.ipynb`: Demonstrates the application of HGNN on sc_GEM datasets, including data loading, model training with adjacency features, and downstream analysis.
- `hgnn_PEA_STA.ipynb`: Example of using HGNN for PEA-STA data processing, covering hypergraph construction from adjacency matrices and model evaluation.

To run the notebooks:

```bash
conda activate py39
jupyter notebook
```

Then navigate to the desired notebook file and run the cells sequentially.

### Running Experiments

Use the provided script to run experiments with adjacency features:

```bash
conda activate py39
python train.py --feature adj --random-walk False
```

### Command Line Arguments

Key arguments for customization (with adj feature focus):

- `--data`: Dataset name (default: 'drug')
- `--dimensions`: Number of embedding dimensions (default: 64)
- `--feature`: Features used (set to 'adj' for adjacency matrix, default: 'adj')
- `--random-walk`: Disable random walk (set to False, default: False)
- `--loss`: Loss function type (choices: 'bce', 'mse', 'zinb', 'rank', 'gauss', 'nb')
- `--batch_size`: Training batch size (default: 96)
- `--rw`: Weight of adjacency matrix reconstruction loss (default: 0.01)

For full list of arguments, run:
```bash
python train.py --help
```

## Model Architecture

The HGNN model focuses on hyperedge prediction using adjacency matrix features, with the following key components:
1. Hyperedge prediction model leveraging adjacency features
2. Support for multiple loss functions to optimize hypergraph structure reconstruction

## Evaluation Metrics

The framework supports various evaluation metrics:
- Accuracy
- ROC-AUC
- Average Precision
- Pearson/Spearman correlation coefficients

## Acknowledgments

This project is based on the SCOIT project. We sincerely thank the SCOIT development team for their foundational work that enabled this implementation.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
