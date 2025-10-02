import numpy as np
import pandas as pd
from scoit import sc_multi_omics
import time
from scipy.sparse import csr_matrix
import seaborn as sns
from enum import Enum, unique
from dataclasses import dataclass


@unique
class NodeType(Enum):
    OMICS = 0
    GENES = 1
    CELLS = 2


@dataclass
class HypergraphNode:
    index: int
    type: 'NodeType'
    name: str = None


class MultiOmicsHypergraph:

    def __init__(self, data, feature_names=None, omics_names=None, verbose=True):
        assert all(x.shape[0] == data[0].shape[0] for x in data)
        assert len(data) >= 2

        self.data = data
        self.num_cells = data[0].shape[0]
        self.num_omics = len(data)
        self.num_genes = sum(x.shape[1] for x in data) # Union of features.

        self.num_nodes = sum([self.num_cells, self.num_genes, self.num_omics])
        self.num_edges = sum(np.count_nonzero(x) for x in data)

        # Omics, Genes, Cells
        self.omics_offset = 0
        self.genes_offset = self.num_omics
        self.cells_offset = self.num_omics + self.num_genes

        if feature_names is not None:
            assert len(feature_names) == self.num_omics
            assert all(x.shape[1] == len(y) for x, y in zip(data, feature_names))
            self.feature_names = sum(feature_names, [])
        else:
            self.feature_names = None

        if omics_names is not None:
            assert len(omics_names) == self.num_omics
            self.omics_names = omics_names
        else:
            self.omics_names = None

    def __repr__(self):
        clsname = self.__class__.__name__
        return f'<{clsname}(#Nodes={self.num_nodes}, #Edges={self.num_edges}, #Omics={self.num_omics}, #Cells={self.num_cells}, #Genes={self.num_genes})>'

    @property
    def edge_rate(self):
        return self.num_edges / self.num_nodes ** 2

    def get_node_type(self, node_id: int):
        if self.omics_offset <= node_id < self.genes_offset:
            return NodeType.OMICS
        if self.genes_offset <= node_id < self.cells_offset:
            return NodeType.GENES
        if self.cells_offset <= node_id < self.num_nodes:
            return NodeType.CELLS
        raise IndexError(node_id)

    def get_node_name(self, node_id: int):
        node_type = self.get_node_type(node_id)
        if node_type == NodeType.OMICS:
            return self.omics_names[node_id - self.omics_offset]
        if node_type == NodeType.GENES:
            return self.feature_names[node_id - self.genes_offset]
        if node_type == NodeType.CELLS:
            return f'Cell_{node_id - self.cells_offset}'
        raise TypeError(node_type)

    def nodes(self):
        for node_id in range(self.num_nodes):
            if self.omics_names and self.feature_names:
                node_name = self.get_node_name(node_id)
            else:
                node_name = None
            yield HypergraphNode(index=node_id, type=self.get_node_type(node_id),
                                 name=node_name)

    def edges(self, return_dict=False):
        for omics_id in range(self.num_omics):
            omics_node = omics_id + self.omics_offset
            feature_matrix = self.data[omics_id]
            for row, col in zip(*np.nonzero(feature_matrix)):
                weight = feature_matrix[row, col]
                # Row is cell id, col is gene id. Convert to node id
                cell_node = row + self.cells_offset
                gene_node = col + self.genes_offset
                if return_dict:
                    yield dict(omics_node=omics_node, cell_node=cell_node, gene_node=gene_node, weight=weight)
                else:
                    yield omics_node, cell_node, gene_node, weight

    def edges_dataframe(self):
        return pd.DataFrame.from_records(self.edges(return_dict=True))


    def to_hypernetx_graph(self):
        import hypernetx as hnx

        hg = hnx.Hypergraph([e[:-1] for e in self.edges()])

        return hg



if __name__ == '__main__':
    expression_data = np.array(pd.read_csv("data/sc_GEM/expression_data.csv", index_col=0))
    methylation_data = np.array(pd.read_csv("data/sc_GEM/methylation_data.csv", index_col=0))
    data = [expression_data, methylation_data]
    mohg = MultiOmicsHypergraph(data)
    print(mohg)
    print(mohg.edge_rate)

    hg = mohg.to_hypernetx_graph()
    # print(hg)
    hg.add
    print('#Edges', hg.number_of_edges(), '#Nodes', hg.number_of_nodes())
    print(hg.incidence_matrix())
    exit()

    import hypernetx as hnx
    import matplotlib.pyplot as plt
    hnx.draw(hg, layout='matrix')
    plt.savefig('H.png')
