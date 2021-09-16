import pytest

def test_idx_permutation_equivariance():
    import numpy as np
    import torch
    permutation_idx = np.random.permutation(5)
    print(permutation_idx)
    permutation_matrix = np.zeros((5, 5), dtype=np.float32)
    permutation_matrix[
        np.arange(5),
        permutation_idx,
    ] = 1
    permutation_matrix = torch.tensor(permutation_matrix, dtype=torch.float32)

    import dgl
    g0 = dgl.rand_graph(5, 10)
    a1 = (permutation_matrix @ g0.adj().to_dense()).type(torch.int32)
    print(a1.shape)
    a1 = a1.to_sparse()
    g1 = dgl.graph(
        data=(a1.indices()[0, :], a1.indices()[1, :]),
        num_nodes=5,
    )

    assert (permutation_matrix @ g0.adj().to_dense()\
        == g1.adj().to_dense()).prod()

    import hpno
    # g0 = hpno.heterograph(g0)
    # g1 = hpno.heterograph(g1)

    print(g0.adj().to_dense())
    print(g1.adj().to_dense())

    h0 = torch.randn(5, 3)
    h1 = permutation_matrix @ h0

    layer = dgl.nn.GraphConv(3, 5)
    print(layer(g0, h0))
    print(layer(g1, h1))

    # layer = hpno.HierarchicalPathNetworkLayer(3, 4, 5, max_level=2)
    # print(layer(g0, h0))
    # print(layer(g1, h1))
