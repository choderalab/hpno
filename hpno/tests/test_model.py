import pytest

def test_ethane_model():
    import torch
    import hpno
    import dgllife
    g = dgllife.utils.smiles_to_bigraph("CC", explicit_hydrogens=True)
    g = hpno.heterograph(g)
    g.nodes['n1'].data['h'] = torch.randn(8, 3)
    model = hpno.HierarchicalPathNetwork(3, 4, 5, 6, activation=torch.nn.Softmax(dim=-1))
    feat = model(g, g.nodes['n1'].data['h'])
    assert feat.shape == torch.Size([8, 4])
