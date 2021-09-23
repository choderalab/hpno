import pytest
import numpy.testing as npt

@pytest.fixture
def graphs_and_features():
    import numpy as np
    import torch
    permutation_idx = np.random.permutation(5)
    permutation_matrix = np.zeros((5, 5), dtype=np.float32)
    permutation_matrix[
        np.arange(5),
        permutation_idx,
    ] = 1
    permutation_matrix = torch.tensor(permutation_matrix, dtype=torch.float32)

    import dgl
    g0 = dgl.rand_graph(5, 20)
    g1 = dgl.reorder_graph(
        g0,
        "custom",
        permute_config={"nodes_perm": permutation_idx}
    )

    import hpno
    g0 = hpno.heterograph(g0)
    g1 = hpno.heterograph(g1)

    h0 = torch.randn(5, 3)
    h1 = permutation_matrix @ h0

    return g0, g1, h0, h1, permutation_matrix

def test_layer_equivariance(graphs_and_features):
    g0, g1, h0, h1, permutation_matrix = graphs_and_features

    import hpno
    layer = hpno.HierarchicalPathNetworkLayer(3, 4, max_level=4)
    y0 = layer(g0, h0)
    y1 = layer(g1, h1)
    npt.assert_almost_equal(
        (permutation_matrix @ y0).detach().numpy(),
        y1.detach().numpy(),
        decimal=5,
    )

def test_model_equivariance(graphs_and_features):
    g0, g1, h0, h1, permutation_matrix = graphs_and_features

    import hpno
    model = hpno.HierarchicalPathNetwork(3, 4, 5, 2, max_level=4)
    y0 = model(g0, h0)
    y1 = model(g1, h1)
    npt.assert_almost_equal(
        (permutation_matrix @ y0).detach().numpy(),
        y1.detach().numpy(),
        decimal=5,
    )

def test_readout_invariance(graphs_and_features):
    g0, g1, h0, h1, permutation_matrix = graphs_and_features

    import hpno
    readout = hpno.GraphReadout(3, 4, 2, max_level=4)
    y0 = readout(g0, h0)
    y1 = readout(g1, h1)
    npt.assert_almost_equal(
        y0.detach().numpy(),
        y1.detach().numpy(),
        decimal=3,
    )

def test_model_and_readout_invariance(graphs_and_features):
    g0, g1, h0, h1, permutation_matrix = graphs_and_features

    import hpno
    readout = hpno.HierarchicalPathNetwork(
        3, 4, 5, 2,
        max_level=4,
        readout=hpno.GraphReadout(4, 4, 6)
    )

    y0 = readout(g0, h0)
    y1 = readout(g1, h1)
    npt.assert_almost_equal(
        y0.detach().numpy(),
        y1.detach().numpy(),
        decimal=3,
    )
