import pytest
import pandas as pd
from pathlib import Path
import json
import numpy as np
from .hypergraph import process_data, load_graph, HyperGraph


@pytest.fixture
def temp_data(tmp_path):
    """Fixture to create temporary multi-omics data files"""
    # Create two omics data files
    omics1_data = {
        'cell_id': ['cell1', 'cell2', 'cell3'],
        'geneA': [1.0, np.nan, 3.0],
        'geneB': [4.0, 5.0, np.nan]
    }
    omics2_data = {
        'cell_id': ['cell1', 'cell2', 'cell3'],
        'geneC': [np.nan, 7.0, 8.0],
        'geneD': [9.0, 10.0, 11.0]
    }

    # Save to temporary CSV files
    omics1_path = tmp_path / 'omics1.csv'
    omics2_path = tmp_path / 'omics2.csv'

    pd.DataFrame(omics1_data).to_csv(omics1_path, index=False)
    pd.DataFrame(omics2_data).to_csv(omics2_path, index=False)

    return {
        'omics1': omics1_path,
        'omics2': omics2_path
    }

def test_process_and_load_graph(temp_data, tmp_path):
    """Test the complete pipeline: process data and load the graph"""
    # Configuration parameters
    cell_key = 'cell_id'
    dataname = 'test_dataset'
    prefix = tmp_path

    # Process the data
    process_data(
        multi_omics_data=temp_data,
        cell_key=cell_key,
        dataname=dataname,
        prefix=prefix,
        fast_edge=True
    )

    # Verify output files exist
    output_dir = prefix / dataname
    assert output_dir.exists()
    assert (output_dir / 'nodes.csv').exists()
    assert (output_dir / 'edges.csv').exists()
    assert (output_dir / 'metadata.json').exists()

    # Load the graph
    graph = load_graph(output_dir, for_test=True)

    # Verify graph is a HyperGraph instance
    assert isinstance(graph, HyperGraph)

    # Check metadata consistency
    with open(output_dir / 'metadata.json', 'r') as f:
        metadata = json.load(f)

    assert graph.num_nodes == metadata['num_nodes']
    assert graph.num_edges == metadata['num_edges']

    # Verify node counts match expectations
    # 2 omics nodes + 4 genes + 3 cells = 9 nodes
    assert graph.num_nodes == 2 + 4 + 3  # 9

    # Verify edge count (count non-NaN values from input data)
    # omics1 has 4 non-NaNs, omics2 has 5 non-NaNs = 9 edges
    assert graph.num_edges == 9

    # Verify node types
    omics_nodes = [n for n in graph.nodes if n.type == 0]
    gene_nodes = [n for n in graph.nodes if n.type == 1]
    cell_nodes = [n for n in graph.nodes if n.type == 2]

    assert len(omics_nodes) == 2
    assert len(gene_nodes) == 4
    assert len(cell_nodes) == 3

    # Verify edge structure
    for edge in graph.edges:
        assert len(edge.nodes) == 3  # Each edge connects 3 nodes (omics, gene, cell)
        assert isinstance(edge.weight, (int, float))

    # Run built-in consistency check
    graph.check()

def test_process_data_with_different_options(temp_data, tmp_path):
    """Test processing with different parameters"""
    # Test with fast_edge=False
    process_data(
        multi_omics_data=temp_data,
        cell_key='cell_id',
        dataname='test_slow',
        prefix=tmp_path,
        fast_edge=False
    )

    # Test with custom read_csv_args
    process_data(
        multi_omics_data=temp_data,
        cell_key='cell_id',
        dataname='test_custom_csv',
        prefix=tmp_path,
        read_csv_args={'sep': ','}
    )

    # Verify outputs exist
    assert (tmp_path / 'test_slow' / 'edges.csv').exists()
    assert (tmp_path / 'test_custom_csv' / 'edges.csv').exists()

@pytest.fixture
def controlled_data(tmp_path):
    """Create controlled multi-omics data with known values for validation"""
    # Create two omics datasets with explicit known values (including NaNs)
    omics_rna_data = {
        'cell_id': ['c1', 'c2', 'c3'],  # 3 cells
        'BRCA1_rna': [2.5, np.nan, 3.8],  # Explicit omics suffix
        'TP53_rna': [1.2, 4.0, np.nan]
    }

    omics_atac_data = {
        'cell_id': ['c1', 'c2', 'c3'],  # Same cells
        'BRCA1_atac': [np.nan, 5.2, 6.1],
        'TP53_atac': [7.3, 8.9, 9.4]
    }

    # Save to temporary files
    rna_path = tmp_path / 'rna.csv'
    atac_path = tmp_path / 'atac.csv'

    pd.DataFrame(omics_rna_data).to_csv(rna_path, index=False)
    pd.DataFrame(omics_atac_data).to_csv(atac_path, index=False)

    # Return data paths and original dataframes for comparison
    return {
        'data_paths': {'rna': rna_path, 'atac': atac_path},
        'original_rna': pd.DataFrame(omics_rna_data),
        'original_atac': pd.DataFrame(omics_atac_data)
    }

def test_edge_weights_match_original_data(controlled_data, tmp_path):
    """Verify that hypergraph edges exactly match original multi-omics values"""
    # Unpack test data
    data_paths = controlled_data['data_paths']
    original_rna = controlled_data['original_rna']
    original_atac = controlled_data['original_atac']

    # Process the data into a hypergraph
    process_data(
        multi_omics_data=data_paths,
        cell_key='cell_id',
        dataname='validation_test',
        prefix=tmp_path,
        fast_edge=True  # Test fast path - we'll test slow path separately
    )

    # Load the constructed graph
    graph = load_graph(tmp_path / 'validation_test', for_test=True)
    metadata = graph.metadata

    # Get critical offsets from metadata
    omics_offset = metadata['omics_offset']
    gene_offset = metadata['gene_offset']
    cell_offset = metadata['cell_offset']
    num_omics = metadata['num_omics']

    # Create mapping from node names to IDs for lookup
    node_name_to_id = {node.name: node.id for node in graph.nodes}

    # --- Verify RNA omics edges ---
    # Check each gene in RNA data
    for gene in ['BRCA1_rna', 'TP53_rna']:
        # Get original values for this gene across cells
        original_values = original_rna[['cell_id', gene]].dropna()

        for _, row in original_values.iterrows():
            cell_name = row['cell_id']
            expected_weight = row[gene]

            # Find corresponding node IDs
            omics_node_id = node_name_to_id['Omics_rna']
            gene_node_id = node_name_to_id[gene + '_rna'] # Adjusted for renamed columns
            cell_node_id = node_name_to_id[cell_name]

            # Find matching edge in graph
            matching_edges = [
                edge for edge in graph.edges
                if edge.nodes == [omics_node_id, gene_node_id, cell_node_id]
            ]

            # Verify exactly one edge exists with correct weight
            assert len(matching_edges) == 1, \
                f"Missing edge for {gene} in {cell_name}"
            assert matching_edges[0].weight == expected_weight, \
                f"Weight mismatch for {gene} in {cell_name}: " \
                f"expected {expected_weight}, got {matching_edges[0].weight}"

    # --- Verify ATAC omics edges ---
    for gene in ['BRCA1_atac', 'TP53_atac']:
        original_values = original_atac[['cell_id', gene]].dropna()

        for _, row in original_values.iterrows():
            cell_name = row['cell_id']
            expected_weight = row[gene]

            omics_node_id = node_name_to_id['Omics_atac']
            gene_node_id = node_name_to_id[gene + '_atac'] # Adjusted for renamed columns
            cell_node_id = node_name_to_id[cell_name]

            matching_edges = [
                edge for edge in graph.edges
                if edge.nodes == [omics_node_id, gene_node_id, cell_node_id]
            ]

            assert len(matching_edges) == 1, \
                f"Missing edge for {gene} in {cell_name}"
            assert matching_edges[0].weight == expected_weight, \
                f"Weight mismatch for {gene} in {cell_name}"

def test_nan_values_are_excluded(controlled_data, tmp_path):
    """Verify that NaN values in original data are not included as edges"""
    data_paths = controlled_data['data_paths']
    original_rna = controlled_data['original_rna']
    original_atac = controlled_data['original_atac']

    # Process with slow edge construction to test both paths
    process_data(
        multi_omics_data=data_paths,
        cell_key='cell_id',
        dataname='nan_test',
        prefix=tmp_path,
        fast_edge=False
    )

    graph = load_graph(tmp_path / 'nan_test', for_test=True)
    node_name_to_id = {node.name: node.id for node in graph.nodes}

    # Check RNA NaNs are excluded
    rna_nan_positions = original_rna[original_rna.isna().any(axis=1)]
    for gene in ['BRCA1_rna', 'TP53_rna']:
        for _, row in rna_nan_positions.iterrows():
            if pd.isna(row[gene]):
                cell_name = row['cell_id']

                # Check this (cell, gene) pair has no edge
                gene_node_id = node_name_to_id.get(gene)
                cell_node_id = node_name_to_id.get(cell_name)
                omics_node_id = node_name_to_id.get('Omics_rna')

                if all([gene_node_id, cell_node_id, omics_node_id]):
                    matching_edges = [
                        edge for edge in graph.edges
                        if edge.nodes == [omics_node_id, gene_node_id, cell_node_id]
                    ]
                    assert len(matching_edges) == 0, \
                        f"Found edge for NaN value: {gene} in {cell_name}"

    # Check ATAC NaNs are excluded
    atac_nan_positions = original_atac[original_atac.isna().any(axis=1)]
    for gene in ['BRCA1_atac', 'TP53_atac']:
        for _, row in atac_nan_positions.iterrows():
            if pd.isna(row[gene]):
                cell_name = row['cell_id']

                gene_node_id = node_name_to_id.get(gene)
                cell_node_id = node_name_to_id.get(cell_name)
                omics_node_id = node_name_to_id.get('Omics_atac')

                if all([gene_node_id, cell_node_id, omics_node_id]):
                    matching_edges = [
                        edge for edge in graph.edges
                        if edge.nodes == [omics_node_id, gene_node_id, cell_node_id]
                    ]
                    assert len(matching_edges) == 0, \
                        f"Found edge for NaN value: {gene} in {cell_name}"

def test_node_offsets_correctness(controlled_data, tmp_path):
    """Verify node offsets in metadata match actual node IDs"""
    process_data(
        multi_omics_data=controlled_data['data_paths'],
        cell_key='cell_id',
        dataname='offset_test',
        prefix=tmp_path
    )

    graph = load_graph(tmp_path / 'offset_test', for_test=True)
    metadata = graph.metadata

    # Verify offset calculations
    assert metadata['omics_offset'] == 0
    assert metadata['gene_offset'] == metadata['num_omics']  # 2 omics
    assert metadata['cell_offset'] == metadata['num_omics'] + metadata['num_genes']  # 2 + 4

    # Verify first cell ID matches cell offset
    first_cell_node = next(n for n in graph.nodes if n.type == 2)
    assert first_cell_node.id == metadata['cell_offset']

    # Verify first gene ID matches gene offset
    first_gene_node = next(n for n in graph.nodes if n.type == 1)
    assert first_gene_node.id == metadata['gene_offset']

