import pytest

def test_ethane():
    import torch
    import hpno
    import dgllife
    g = dgllife.utils.smiles_to_bigraph("CC", explicit_hydrogens=True)
    g = hpno.heterograph(g)
    g.nodes['n1'].data['h'] = torch.zeros(8, 3)
    layer = hpno.HierarchicalPathNetworkLayer(3, 4, 5)
    layer(g, g.nodes['n1'].data['h'])
