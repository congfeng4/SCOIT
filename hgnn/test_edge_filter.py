import pytest
import tempfile
from pathlib import Path
import pandas as pd
from .hypergraph import HyperGraphCreator

@pytest.fixture(scope="function")
def temp_dir():
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)

def create_test_csv(temp_dir, data, filename="data.csv"):
    path = temp_dir / filename
    pd.DataFrame(data).to_csv(path, index=False)
    return path

def test_non_negative_filter(temp_dir):
    """Test non-negative filter (> 0) with appropriate data"""
    # Data: non-negative values (including zeros) - should trigger >0 filter
    data = {
        "cell_id": ["c1", "c2", "c3"],
        "gene1": [0, 2.5, 0],    # Zeros should be filtered
        "gene2": [3.0, 0, 5.0]   # Zeros should be filtered
    }
    data_path = create_test_csv(temp_dir, data)

    # Run processor with filtering enabled
    creator = HyperGraphCreator(
        multi_omics_data={"omics": data_path},
        cell_key="cell_id",
        dataname="test",
        prefix=temp_dir,
        filter_edge=True,
        fast_edge=True
    )
    creator.process()

    # Load results
    edges_df = pd.read_csv(creator.savedir / "edges.csv")

    # Verify filter type
    assert creator.filter_edge == "non-negative"

    # Verify results: only 2.5, 3.0, 5.0 should remain (3 edges)
    assert len(edges_df) == 3
    assert set(edges_df["Weight"]) == {2.5, 3.0, 5.0}
    assert 0.0 not in edges_df["Weight"].values
