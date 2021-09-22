import pytest

def test_ethane_layer():
    import torch
    import hpno
    import dgllife
    g = dgllife.utils.smiles_to_bigraph("CC", explicit_hydrogens=True)
    g = hpno.heterograph(g)
    g.nodes['n1'].data['h'] = torch.zeros(8, 3)
    layer = hpno.HierarchicalPathNetworkLayer(3, 4)
    feat = layer(g, g.nodes['n1'].data['h'])
    assert feat.shape == torch.Size([8, 4])
