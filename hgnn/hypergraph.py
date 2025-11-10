from pathlib import Path
import numpy as np
import pandas as pd
import time
from dataclasses import dataclass
import json
from collections import Counter, defaultdict
from typing import Union
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split


def robust_zscore(x):
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    mad_corr = mad / 0.6745
    # 避免 mad=0（所有值相同）
    mad_corr = np.clip(mad_corr, a_min=1e-8, a_max=None)
    return (x - med) / mad_corr


def robust_zscore_filter(series, thresh=3.5):
    rz = robust_zscore(series.dropna())
    mask = np.abs(rz) <= thresh
    return series.where(mask, np.nan)  # 异常值置 NaN

def minmax_normalize(df: pd.DataFrame):
    """
    Min-Max normalize a DataFrame **without touching** index or columns.
    Returns a new DataFrame with the same shape, index and columns.
    """
    scaler = MinMaxScaler()
    # keep the order: index → columns → values
    normed = scaler.fit_transform(df.values)
    return pd.DataFrame(normed, index=df.index, columns=df.columns)


def save_edges_and_weights(save_dir, edges, weights, train_size, normalization):
    # Normalize 会让本来没有0的数据集产生零，即本来的最小值会变成0
    save_dir = Path(save_dir)
    nums_type = edges.max(0) + 1

    weights = weights.reshape(-1, 1)

    if normalization:
        scaler = MinMaxScaler()
        weights = scaler.fit_transform(weights)
        print('Normalize', scaler)

    train_data, test_data, train_weight, test_weight = train_test_split(edges, weights, train_size=train_size)
    train_weight = train_weight.reshape(-1)
    test_weight = test_weight.reshape(-1)

    np.savez(save_dir / 'train_data.npz', train_data=train_data, train_weight=train_weight,
             nums_type=nums_type)
    np.savez(save_dir / 'test_data.npz', test_data=test_data, test_weight=test_weight,
             nums_type=nums_type)


def save_metadata(save_dir, edges, train_size, nums_type, normalization):
    metadata = dict(
        train_size=train_size,
        normalization=normalization,
        num_edges=edges.shape[0],
        num_nodes=sum(nums_type),
        num_cells=nums_type[0],
        num_genes=nums_type[1],
        num_omics=nums_type[2],
    )
    np.savez(save_dir / 'metadata', **metadata)


class HyperGraphCreator:
    """
    A class to process multi-omics data into a hypergraph and save it to disk.

    Each omics data is a CSV file with rows as cells and columns as genes.
    The cell_key is the column name of cell identifiers, which should be present in all files
    and will be used to join the dataframes.

    Note: filter_edge=True will cause isolated ndoes, which may prevent training!!
    """

    def __init__(self, multi_omics_data: dict[str, Path], cell_key: str, dataname: str,
                 prefix: Path, filter_edge=False, fast_edge=True, pickle=False, remove_zeros=True,
                 read_csv_args=None):
        """
        Initialize the HyperGraphCreator with processing parameters.

        :param multi_omics_data: A dictionary mapping omics names to their CSV file paths.
        :param cell_key: The column name for cell identifiers.
        :param dataname: The name of the dataset.
        :param prefix: The directory to save the processed hypergraph.
        :param filter_edge: Whether to filter edges based on value distribution.
        :param fast_edge: Whether to use the fast method to construct hyper edges.
        :param pickle: Whether to save edges in pickle format.
        :param read_csv_args: Additional arguments to pass to pd.read_csv.
        """
        if read_csv_args is None:
            read_csv_args = {}
        self.multi_omics_data = multi_omics_data
        self.cell_key = cell_key
        self.dataname = dataname
        self.prefix = Path(prefix)
        self.filter_edge = filter_edge
        self.fast_edge = fast_edge
        self.pickle = pickle
        self.read_csv_args = read_csv_args.copy()

        # Initialize variables to be populated during processing
        self.multi_omics_df = None
        self.df = None
        self.num_omics = len(multi_omics_data)
        self.num_cells = 0
        self.num_genes = 0
        self.num_nodes = 0
        self.omics_offset = 0
        self.gene_offset = 0
        self.cell_offset = 0
        self.savedir = self.prefix / dataname
        self.remove_zeros = remove_zeros

    def get_omics_id_from_gene(self, gene_name: str) -> int:
        """Get the omics identifier from a gene name."""
        for omics_id, omics_name in enumerate(self.multi_omics_data.keys()):
            if gene_name.endswith(omics_name):
                return omics_id
        raise ValueError(f"{gene_name} has no associated omics!")

    def load_and_preprocess_data(self) -> None:
        """Load the multi-omics data and perform initial preprocessing."""
        print('Begin to load multi omics data', self.multi_omics_data)
        self.read_csv_args.update(index_col=self.cell_key)

        begin = time.time()
        self.multi_omics_df = {
            name: pd.read_csv(path, **self.read_csv_args)#.sort_index()
            for name, path in self.multi_omics_data.items()
        }
        print('Load data done, time:', time.time() - begin)

        # Rename feature columns with omics suffix
        self.multi_omics_df = {
            name: df.rename({col: f"{str(col).strip()}_{name}" for col in df.columns}, axis=1)
            for name, df in self.multi_omics_df.items()
        }

        # Join dataframes by cell key
        print('Union features from multi-omics')
        self.df = pd.concat(list(self.multi_omics_df.values()), axis=1)
        print('Columns', len(self.df.columns), 'Cells', len(self.df))

        if self.remove_zeros:
            self.df[self.df == 0] = np.nan
        # Remove all-NaN columns
        self.df.dropna(axis=1, how='all', inplace=True)
        print('Columns after clean', len(self.df.columns), 'Cells', len(self.df))

        # Set counts and offsets
        self.num_cells = len(self.df)
        self.num_genes = len(self.df.columns)
        self.num_nodes = self.num_cells + self.num_genes + self.num_omics

        self.omics_offset = 0
        self.gene_offset = self.num_omics
        self.cell_offset = self.num_omics + self.num_genes
        self.orig_edge_count = 0

        print(f'{self.num_omics=}, {self.num_cells=}, {self.num_genes=}, {self.num_nodes=}')

    def load_cell_stage(self):
        path = Path(list(self.multi_omics_data.values())[0]).with_name('cell_stage.json')
        try:
            self.cell_stage = json.load(path.open('r'))
        except:
            print('No cell stage in', path)
            self.cell_stage = None

    def create_nodes_file(self) -> None:
        """Create and save the nodes CSV file."""

        def add_node_type(nodes, node_type):
            return [(node, node_type) for node in nodes]

        omics_names = [f'Omics_{name}' for name in self.multi_omics_data.keys()]
        cell_names = self.df.index.values.tolist()
        gene_names = list(self.df.columns)

        node_names = (add_node_type(omics_names, 0) +
                      add_node_type(gene_names, 1) +
                      add_node_type(cell_names, 2))

        df_nodes = pd.DataFrame(node_names, columns=['Name', 'Type'])

        # Create save directory if it doesn't exist
        self.savedir.mkdir(parents=True, exist_ok=True)

        node_file = self.savedir / 'nodes.csv'
        df_nodes.to_csv(node_file, index=True)
        print(f'Save node file: {node_file}')

    def construct_hyper_edge(self) -> pd.DataFrame:
        """Construct hyper edges using the standard method."""
        hyper_edges = []
        for cell_id in range(self.num_cells):
            for gene_id in range(self.num_genes):
                weight = self.df.iloc[cell_id, gene_id]
                if np.isnan(weight):
                    continue

                omics_id = self.get_omics_id_from_gene(self.df.columns[gene_id])
                omics_node = omics_id + self.omics_offset
                gene_node = gene_id + self.gene_offset
                cell_node = cell_id + self.cell_offset

                edge = {
                    'Omics': omics_node,
                    'Gene': gene_node,
                    'Cell': cell_node,
                    'Weight': weight
                }
                hyper_edges.append(edge)

        return pd.DataFrame.from_records(hyper_edges)

    def construct_hyper_edge_fast(self) -> pd.DataFrame:
        """Construct hyper edges using a faster vectorized method."""
        # Precompute gene_id → omics_id mapping
        gene2omics = pd.Series(
            [self.get_omics_id_from_gene(g) for g in self.df.columns],
            index=pd.RangeIndex(len(self.df.columns))
        )

        # Replace column names with integers and stack
        df_int_cols = self.df.set_axis(range(self.df.shape[1]), axis=1)
        w = df_int_cols.to_numpy()
        cell_id, gene_id = np.nonzero(~np.isnan(w))  # Get coordinates of non-NaN values
        weight = w[cell_id, gene_id]

        # Create base dataframe
        long = pd.DataFrame({
            'cell_id': cell_id,
            'gene_id': gene_id,
            'Weight': weight
        })

        # Calculate node IDs vectorized
        long['Omics'] = gene2omics[long['gene_id']].values + self.omics_offset
        long['Gene'] = long['gene_id'].values + self.gene_offset
        long['Cell'] = long['cell_id'].values + self.cell_offset

        return long[['Omics', 'Gene', 'Cell', 'Weight']]

    def filter_edges(self) -> None:
        """Filter edges based on value distribution."""
        self.orig_edge_count = self.df.count().sum()
        if self.filter_edge:
            if (self.df >= 0).all().all():
                self.filter_edge = 'non-negative'
                print('All values are non-negative, using > 0 to filter edges')
                self.df[self.df <= 0] = np.nan
            else:
                self.filter_edge = 'robust-zscore'
                print('Values contain negatives, using robust zscore to filter edges')
                for col in self.df.columns:
                    self.df[col] = robust_zscore_filter(self.df[col])

        print(f'Original edges: {self.orig_edge_count}')
        print(f'{self.filter_edge=}, remaining {self.df.count().sum()} edges')

    def save_metadata_and_edges(self, df_edges: pd.DataFrame) -> None:
        """Save metadata and edge information to files."""
        # Create metadata dictionary
        metadata = {
            'dataname': self.dataname,
            'filter_edge': self.filter_edge,
            'num_cells': self.num_cells,
            'num_genes': self.num_genes,
            'num_omics': self.num_omics,
            'num_nodes': self.num_nodes,
            'orig_edges': int(self.orig_edge_count),
            'num_edges': len(df_edges),
            'omics_offset': self.omics_offset,
            'gene_offset': self.gene_offset,
            'cell_offset': self.cell_offset,
            'omics_dims': [len(df.columns) for df in self.multi_omics_df.values()],
            'raw_data': {k: str(v) for k, v in self.multi_omics_data.items()},
            'cell_key': self.cell_key,
        }

        # Save metadata
        metadata_file = self.savedir / 'metadata.json'
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=4)
        print(f'Save metadata: {metadata_file}')

        # Save edges
        edge_file = self.savedir / 'edges.csv'
        if self.pickle:
            edge_file = edge_file.with_suffix('.pkl')
            df_edges.to_pickle(edge_file)
        else:
            df_edges.to_csv(edge_file, index=False)
        print(f'Save edge file: {edge_file}')

    def process(self) -> None:
        """Execute the complete hypergraph creation pipeline."""
        self.load_and_preprocess_data()
        self.create_nodes_file()
        self.filter_edges()

        begin = time.time()
        print('Constructing hyper edges...')
        if self.fast_edge:
            df_edges = self.construct_hyper_edge_fast()
        else:
            df_edges = self.construct_hyper_edge()

        assert check_isolated_nodes(df_edges, num_nodes=self.num_nodes)
        print('Construct edges done, time:', time.time() - begin)

        self.save_metadata_and_edges(df_edges)


def process_data(multi_omics_data: dict[str, Path], cell_key: str, dataname: str, prefix: Path, filter_edge=False,
                 fast_edge=True,
                 pickle=False,
                 read_csv_args: dict = {}):
    """
    Process multi-omics data into a hypergraph and save to disk.
    Each omics data is a CSV file with rows as cells and columns as genes.
    The cell_key is the column name of cell identifiers, which should be present in all files
    and will be used to join the dataframes.

    :param multi_omics_data: A dictionary mapping omics names to their CSV file paths.
    :param cell_key: The column name for cell identifiers.
    :param dataname: The name of the dataset.
    :param prefix: The directory to save the processed hypergraph.
    :param filter_edge: Whether to filter edges based on value distribution.
    :param fast_edge: Whether to use the fast method to construct hyper edges.
    :param pickle: Whether to save edges in pickle format.
    :param read_csv_args: Additional arguments to pass to pd.read_csv.
    :return: HyperGraphCreator instance after processing.
    """
    creator = HyperGraphCreator(
        multi_omics_data=multi_omics_data,
        cell_key=cell_key,
        dataname=dataname,
        prefix=prefix,
        filter_edge=filter_edge,
        fast_edge=fast_edge,
        pickle=pickle,
        read_csv_args=read_csv_args
    )
    creator.process()
    return creator


@dataclass
class HyperNode:
    id: int
    name: str
    type: int


@dataclass
class HyperEdge:
    id: int
    nodes: list[int]
    weight: float


@dataclass
class HyperGraph:
    edges: pd.DataFrame
    nodes: pd.DataFrame
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

    @property
    def num_cells(self):
        return self.metadata['num_cells']

    @property
    def num_genes(self):
        return self.metadata['num_genes']

    def __str__(self):
        return f'HyperGraph(num_nodes={self.num_nodes}, num_edges={self.num_edges})'

    def __repr__(self):
        return str(self.metadata)

    @property
    def cell_names(self):
        return self.nodes['Name'][self.metadata['cell_offset'] + np.arange(self.num_cells)]

    @property
    def gene_names(self):
        return self.nodes['Name'][self.metadata['gene_offset'] + np.arange(self.num_genes)]


def load_graph(dir_path: Path, for_test=False, load_edges=True) -> HyperGraph:
    """
    Load a hypergraph from a directory.
    The directory should contain 'nodes.csv', 'edges.csv' (or 'edges.pkl'),
    and 'metadata.json' files.
    :param dir_path: The directory path.
    :return: A HyperGraph object.
    """
    dir_path = Path(dir_path)
    assert dir_path.exists(), f'Path {dir_path} does not exist.'

    with open(dir_path / 'metadata.json', 'r') as f:
        metadata = json.load(f)

    if load_edges:
        begin = time.time()
        try:
            df_edges = pd.read_csv(dir_path / 'edges.csv')
        except FileNotFoundError:
            df_edges = pd.read_pickle(dir_path / 'edges.pkl')
        print(f'Load edges in {time.time() - begin:.2f} seconds.')
    else:
        df_edges = None
        print('load_edges=False')

    begin = time.time()
    df_nodes = pd.read_csv(dir_path / 'nodes.csv', index_col=0)
    print(f'Load nodes in {time.time() - begin:.2f} seconds.')
    print(f'Num nodes: {len(df_nodes)}, Num edges: {len(df_edges) if df_edges is not None else "NaN"}')

    if for_test and load_edges:
        nodes = [HyperNode(id=i, type=row.Type, name=row.Name) for i, row in df_nodes.iterrows()]
        edges = [HyperEdge(id=i, nodes=list(map(int, [row.Omics, row.Gene, row.Cell])), weight=float(row.Weight))
                 for i, row in df_edges.iterrows()]
        graph = HyperGraph(nodes=nodes, edges=edges, metadata=metadata)
    else:
        graph = HyperGraph(nodes=df_nodes, edges=df_edges, metadata=metadata)
        if load_edges:
            assert check_isolated_nodes(graph)

    print('Metadata:', metadata)
    return graph


def load_graph_metadata(dir_path: Path):
    """
    Load metadata of all hypergraphs in a directory.
    Each hypergraph should be in a subdirectory with a 'metadata.json' file.
    :param dir_path: The directory path.
    :return: A DataFrame with metadata of all hypergraphs.
    """

    assert dir_path.exists(), f'Path {dir_path} does not exist.'
    records = []

    for subdir in dir_path.iterdir():
        if subdir.is_dir() and (subdir / 'metadata.json').exists():
            print('Found dataset:', subdir.name)
            with open(subdir / 'metadata.json', 'r') as f:
                metadata = json.load(f)
            records.append(metadata)

    df = pd.DataFrame.from_records(records)
    df = df['dataname num_cells num_genes num_omics num_nodes num_edges'.split()]
    return df


def convert_to_hypersagnn_format(hg: HyperGraph, train_size: float, save_dir: Path, normalization=True):
    assert 0 <= train_size <=1, train_size

    edges = hg.edges
    metadata = hg.metadata
    metadata.update(normalization=normalization, train_size=train_size)

    df = pd.DataFrame({
        'Cell': edges['Cell'] - metadata['cell_offset'],
        'Gene': edges['Gene'] - metadata['gene_offset'],
        'Omics': edges['Omics'] - metadata['omics_offset'],
        'Weight': edges['Weight'],
    })
    nums_type = [metadata['num_cells'], metadata['num_genes'], metadata['num_omics']]
    print('nums_type', nums_type)

    edges_data = df[['Cell', 'Gene', 'Omics']].values
    edges_weight = df['Weight'].values
    print('num_edges', edges_data.shape[0])

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    if train_size == 1:
        print('Train set only! No test set!')
        np.savez(save_dir / 'train_data.npz', train_data=edges_data, train_weight=edges_weight,
                 nums_type=nums_type)
        assert check_hypergraph_isolated_nodes(edges_data, nums_type)
        print('Save to', save_dir)
        return

    train_data, test_data, train_weight, test_weight = train_test_split(edges_data, edges_weight,
                                                                          train_size=train_size)
    if normalization:
        scaler = MinMaxScaler()
        train_weight = scaler.fit_transform(train_weight.reshape(-1, 1)).reshape(-1)
        test_weight = scaler.transform(test_weight.reshape(-1, 1)).reshape(-1)
        metadata.update(weight_min=scaler.data_min_[0], weight_max=scaler.data_max_[0])
        print('Normalized, min_val', scaler.data_min_, 'max_val', scaler.data_max_)

    print('train_data', train_data.shape, 'train_weight', train_weight.shape,
          'test_data', test_data.shape, 'test_weight', test_weight.shape)

    assert check_hypergraph_isolated_nodes(train_data, nums_type)
    np.savez(save_dir / 'train_data.npz', train_data=train_data, train_weight=train_weight,
             nums_type=nums_type)

    # assert check_hypergraph_isolated_nodes(test_data, nums_type)
    np.savez(save_dir / 'test_data.npz', test_data=test_data, test_weight=test_weight,
             nums_type=nums_type)

    # Update metadata
    metadata_file = save_dir / 'metadata.json'
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=4)
    print(f'Update metadata: {metadata}')

    print('Save to', save_dir)


def check_isolated_nodes(hg: Union[HyperGraph, pd.DataFrame, np.ndarray],
                          num_nodes: int = None, nums_type: np.array = None):
    if isinstance(hg, HyperGraph):
        node_in_edges = set(hg.edges.values.reshape(-1))
        total_nodes = set(range(hg.num_nodes))
    elif isinstance(hg, pd.DataFrame): # Edge Df
        assert num_nodes, num_nodes
        node_in_edges = set(hg.values.reshape(-1))
        total_nodes = set(range(num_nodes))
    elif isinstance(hg, np.ndarray):  # Hyper-SAGNN format
        assert nums_type, nums_type
        return check_hypergraph_isolated_nodes(hg, nums_type)

    isolated_nodes = total_nodes - node_in_edges
    if isolated_nodes:
        print('Detected isolated ndoes', isolated_nodes)
    return not isolated_nodes


def check_hypergraph_isolated_nodes(E: np.array, nums_type: np.array):
    # nums_type: how many nodes for each type.

    node_in_edges = defaultdict(set)
    for i in range(E.shape[0]): # All edges.
        for j in range(E.shape[1]): # Each node in an edge.
            node_in_edges[j].add(E[i, j]) # One edge mentions this node.

    # If no edge or just one edge mentions a node, then the node is isolated (no neighbours).
    isolated_nodes = []
    for node_type, num_nodes in enumerate(nums_type):
        all_nodes = set(range(num_nodes))
        isolated_nodes_ = all_nodes - node_in_edges[node_type]
        if isolated_nodes_:
            isolated_nodes.extend([(node_type, nid) for nid in isolated_nodes_])

    if isolated_nodes:
        print('Detected isolated_nodes', isolated_nodes)

    return not isolated_nodes
