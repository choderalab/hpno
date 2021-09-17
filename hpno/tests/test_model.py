import pytest

def test_ethane_model():
    import torch
    import hpno
    import dgllife
    g = dgllife.utils.smiles_to_bigraph("CC", explicit_hydrogens=True)
    g = hpno.heterograph(g)
    g.nodes['n1'].data['h'] = torch.zeros(8, 3)
    model = hpno.HierarchicalPathNetwork(3, 4, 5, 6)
    feat = model(g, g.nodes['n1'].data['h'])
    assert feat.shape == torch.Size([8, 4])
