import pandas as pd
import numpy as np
import torch
import json
from pathlib import Path
from scipy.sparse import csr_matrix
from .utils import np2tensor_hyper, add_padding_idx, parallel_build_hash  # Reuse from original code


def load_multi_omics_hypergraph(data_dir: str) -> tuple:
    """
    Load multi-omics hypergraph (nodes/edges/metadata) and format for Hyper_SAGNN.

    Args:
        data_dir: Path to directory containing nodes.csv, edges.csv, metadata.json.

    Returns:
        train_data: Tensor of hyperedges (N, 3) [Omics, Gene, Cell] (padded).
        train_weight: Tensor of edge weights (N, 1).
        test_data: Tensor of test hyperedges (M, 3) (padded).
        test_weight: Tensor of test edge weights (M, 1).
        num_type: List of node counts per type [num_omics, num_genes, num_cells].
        num_list: Cumulative node counts (for node ID mapping).
    """
    data_dir = Path(data_dir)

    # 1. Load metadata (get node type counts)
    with open(data_dir / "metadata.json", "r") as f:
        metadata = json.load(f)
    num_omics = metadata["num_omics"]
    num_genes = metadata["num_genes"]
    num_cells = metadata["num_cells"]
    num_type = [num_omics, num_genes, num_cells]
    num_list = np.cumsum(num_type)  # Maps node types to global IDs (e.g., Cells start at num_omics + num_genes)

    # 2. Load edges (train/test split: 80-20 by default)
    edges_df = pd.read_csv(data_dir / "edges.csv")
    edges = edges_df[["Omics", "Gene", "Cell"]].values  # (N, 3) hyperedges
    weights = edges_df["Weight"].values.reshape(-1, 1).astype(np.float32)  # (N, 1) non-negative weights

    # Train-test split (stratify to preserve edge distribution)
    split_idx = int(0.8 * len(edges))
    train_data = edges[:split_idx]
    train_weight = weights[:split_idx]
    test_data = edges[split_idx:]
    test_weight = weights[split_idx:]

    # 3. Map node IDs to global IDs (avoid overlap between types)
    # Omics: 1~num_omics (0 = padding), Genes: num_omics+1~num_omics+num_genes, Cells: num_omics+num_genes+1~total
    train_data[:, 1] += num_omics  # Gene IDs → global IDs
    train_data[:, 2] += num_omics + num_genes  # Cell IDs → global IDs
    test_data[:, 1] += num_omics
    test_data[:, 2] += num_omics + num_genes

    # 4. Add padding index (Hyper_SAGNN uses 0 for padding)
    train_data = add_padding_idx(train_data)  # Ensures edge IDs are ≥1 (0 = padding)
    test_data = add_padding_idx(test_data)

    # 5. Convert to PyTorch tensors (padded for uniform length)
    train_data = np2tensor_hyper(train_data, dtype=torch.long)
    test_data = np2tensor_hyper(test_data, dtype=torch.long)
    train_weight = torch.tensor(train_weight, dtype=torch.float32)
    test_weight = torch.tensor(test_weight, dtype=torch.float32)

    # 6. Build hash dictionaries (for negative sampling in training)
    global train_dict, test_dict
    train_dict = parallel_build_hash(train_data, "build_hash", args=None, num=num_type)
    test_dict = parallel_build_hash(test_data, "build_hash", args=None, num=num_type, initial=train_dict)

    return train_data, train_weight, test_data, test_weight, num_type, num_list

# Example Usage
if __name__ == "__main__":
    train_data, train_weight, test_data, test_weight, num_type, num_list = load_multi_omics_hypergraph(
        data_dir="../multi_omics_hypergraph"  # Replace with your directory
    )
    print(f"Train edges: {train_data.shape}, Test edges: {test_data.shape}")
    print(f"Node counts (Omics, Genes, Cells): {num_type}")
    