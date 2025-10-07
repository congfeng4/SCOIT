from pathlib import Path
import numpy as np
import pandas as pd
from scoit import sc_multi_omics
import time
from scipy.sparse import csr_matrix
import seaborn as sns
from enum import Enum, unique
from dataclasses import dataclass
from functools import reduce
import json


@dataclass
class HyperNode:
    id: int
    type: str
    feature: np.ndarray | None = None


@dataclass
class HyperEdge:
    nodes: list[int]
    weight: float


@dataclass
class HyperGraph:
    edges: list[HyperEdge]
    nodes: list[HyperNode]

    def to_dict(self):
        return {
            'edges': [edge.__dict__ for edge in self.edges],
            'num_nodes': self.num_nodes
        }


def load_graph(dir_path: Path):
    assert dir_path.exists(), f'Path {dir_path} does not exist.'

    with open(dir_path / 'metadata.json', 'r') as f:
        metadata = json.load(f)

    df_edges = pd.read_csv(dir_path / 'edges.csv')
    
    return df_edges, metadata


if __name__ == '__main__':
    process_data({
        'expr': './data/sc_GEM/expression_data.csv',
        'methy': './data/sc_GEM/methylation_data.csv',
    }, cell_key='Cell ID')
