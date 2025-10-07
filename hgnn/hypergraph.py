from pathlib import Path
import numpy as np
import pandas as pd
import time
from dataclasses import dataclass
import json


def process_data(multi_omics_data: dict[str, Path], cell_key: str, dataname: str, prefix: Path, fast_edge=True,
                 read_csv_args: dict = {}):

    def get_omics_id_from_gene(gene_name: str):
        for id, omics_name in enumerate(multi_omics_data.keys()):
            if gene_name.endswith(omics_name):
                return id
        raise ValueError(gene_name + ' has no omics!')

    num_omics = len(multi_omics_data)
    print('Begin to load multi omics data', multi_omics_data)
    read_csv_args.update(index_col=cell_key)

    begin = time.time()
    multi_omics_df = {
        name: pd.read_csv(path, **read_csv_args).sort_index() for name, path in multi_omics_data.items()
    }
    print('Load data done, time:', time.time() - begin)

    # Rename feature columns.
    multi_omics_df = {
        name: df.rename({col: str(col).strip() + "_" + name for col in df.columns}, axis=1)
          for name, df in multi_omics_df.items()
    }
    print('Rename columns')
    for name, df in multi_omics_df.items():
        print('Name', name, 'Columns', df.columns)

    print('Union features from multi-omics')
    # Join the dataframes by cell key.
    df = pd.concat(list(multi_omics_df.values()), axis=1)

    # Remove all-NaNs columns
    df.dropna(axis=1, how='all', inplace=True)
    num_cells = len(df)
    num_genes = len(df.columns)
    num_nodes = num_cells + num_genes + num_omics

    print(f'{num_omics=}, {num_cells=}, {num_genes=}, {num_nodes=}')

    def add_node_type(nodes, node_type):
        return [(node, node_type) for node in nodes]

    omics_names = ['Omics_' + name for name in multi_omics_data.keys()]
    cell_names = df.index.values.tolist()
    gene_names = list(df.columns)
    node_names = add_node_type(omics_names, 0) + add_node_type(gene_names, 1) + \
        add_node_type(cell_names, 2)
    df_nodes = pd.DataFrame(node_names, columns=['Name', 'Type'])

    # Node offsets
    omics_offset = 0
    gene_offset = num_omics
    cell_offset = num_omics + num_genes

    def construct_hyper_edge():
        hyper_edges = []
        for cell_id in range(num_cells):
            for gene_id in range(num_genes):
                weight = df.iloc[cell_id, gene_id]
                if np.isnan(weight):
                    continue

                omics_id = get_omics_id_from_gene(gene_names[gene_id])
                omics_node = omics_id + omics_offset
                gene_node = gene_id + gene_offset
                cell_node = cell_id + cell_offset
                edge = dict(Omics=omics_node, Gene=gene_node,
                            Cell=cell_node, Weight=weight)
                hyper_edges.append(edge)

        df_edges =  pd.DataFrame.from_records(hyper_edges)
        return df_edges

    def construct_hyper_edge_fast():

        # 0. 预计算 gene_id → omics_id
        gene2omics = pd.Series(
            [get_omics_id_from_gene(g) for g in df.columns],
            index=pd.RangeIndex(len(df.columns))   # 0,1,2,...
        )

        # 1. 把列名换成整数，再 stack
        df_int_cols = df.set_axis(range(df.shape[1]), axis=1)   # 关键一步
        w = df_int_cols.to_numpy()
        cell_id, gene_id = np.nonzero(~np.isnan(w))   # 非 NaN 的坐标
        weight = w[cell_id, gene_id]
        long = pd.DataFrame({'cell_id': cell_id,
                     'gene_id': gene_id,
                     'Weight': weight})

        # ---- 2. 三列 node id 向量化 ----
        long['Omics']  = gene2omics[long['gene_id']].values + omics_offset
        long['Gene']   = long['gene_id'].values + gene_offset
        long['Cell']   = long['cell_id'].values + cell_offset

        # ---- 3. 只要这四列 ----
        hyper_edges = long[['Omics', 'Gene', 'Cell', 'Weight']]
        return hyper_edges

    begin = time.time()
    print('Constructing hyper edges...')
    df_edges = construct_hyper_edge_fast() if fast_edge else construct_hyper_edge()
    print('Construct edges done, time:', time.time() - begin)

    savedir = Path(prefix) / dataname
    savedir.mkdir(parents=True, exist_ok=True)
    node_file = savedir / 'nodes.csv'
    df_nodes.to_csv(node_file, index=True)
    print('Save node file', node_file)

    edge_file = savedir / 'edges.csv'
    df_edges.to_csv(edge_file, index=False)
    print('Save edge file', edge_file)

    metadata = dict(
        num_cells=num_cells, num_genes=num_genes, num_omics=num_omics,
        num_nodes=num_nodes, num_edges=len(df_edges),
        omics_offset=omics_offset, gene_offset=gene_offset, cell_offset=cell_offset,
        # Convert Path objects to strings using str()
        raw_data={k: str(v) for k, v in multi_omics_data.items()},
        cell_key=cell_key,
    )

    metadata_file = savedir / 'metadata.json'
    json.dump(metadata, metadata_file.open('w'), indent=4)
    print('Save metadata', metadata_file)



@dataclass
class HyperNode:
    id: int
    name: str
    type: int


@dataclass
class HyperEdge:
    nodes: list[int]
    weight: float


@dataclass
class HyperGraph:
    edges: list[HyperEdge]
    nodes: list[HyperNode]
    metadata: dict

    def check(self):
        assert self.metadata['num_nodes'] == self.num_nodes, \
            f"Metadata num_nodes {self.metadata['num_nodes']} != actual num_nodes {self.num_nodes}"
        assert self.metadata['num_edges'] == self.num_edges, \
            f"Metadata num_edges {self.metadata['num_edges']} != actual num_edges {self.num_edges}"
        print('Hypergraph OK')

    @property
    def num_nodes(self):
        return len(self.nodes)

    @property
    def num_edges(self):
        return len(self.edges)

    def __str__(self):
        return f'HyperGraph(num_nodes={self.num_nodes}, num_edges={self.num_edges})'

    def __repr__(self):
        return str(self.metadata)


def load_graph(dir_path: Path):
    assert dir_path.exists(), f'Path {dir_path} does not exist.'

    with open(dir_path / 'metadata.json', 'r') as f:
        metadata = json.load(f)
    begin = time.time()
    df_edges = pd.read_csv(dir_path / 'edges.csv')
    print(f'Load edges in {time.time() - begin:.2f} seconds.')

    df_nodes = pd.read_csv(dir_path / 'nodes.csv', index_col=0)
    print('Load nodes and edges from', dir_path)
    print(f'Num nodes: {len(df_nodes)}, Num edges: {len(df_edges)}')
    print('Metadata:', metadata)
    nodes = [HyperNode(id=i, type=row.Type, name=row.Name) for i, row in df_nodes.iterrows()]
    edges = [HyperEdge(nodes=list(map(int, [row.Omics, row.Gene, row.Cell])), weight=float(row.Weight))
             for _, row in df_edges.iterrows()]
    graph = HyperGraph(nodes=nodes, edges=edges, metadata=metadata)
    return graph
