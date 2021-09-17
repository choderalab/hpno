import pytest

def test_ethane_readout():
    import torch
    import hpno
    import dgllife
    g = dgllife.utils.smiles_to_bigraph("CC", explicit_hydrogens=True)
    g = hpno.heterograph(g)
    g.nodes['n1'].data['h'] = torch.zeros(8, 3)
    readout = hpno.GraphReadout(3, 4, 5)
    feat = readout(g, g.nodes['n1'].data['h'])
    assert feat.shape == torch.Size([1, 4])
