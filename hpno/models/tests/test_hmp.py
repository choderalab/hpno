import pytest

def test_import():
    import hpno as hq
    hq.models

def test_init():
    import hpno as hq

    net = hq.models.hmp.HierarchicalMessagePassing(units=32)

def test_forward():
    import hpno as hq
    import torch
    from rdkit import Chem

    g = hq.heterograph.from_homograph(
            hq.graph.from_rdkit_mol(
                Chem.MolFromSmiles('c1ccccc1')
            )
    )

    net = torch.nn.Sequential(
        hq.models.input_layer.InputLayer(in_features=117, out_features=32),
        hq.models.hmp.HierarchicalMessagePassing(units=32),
    )

    g = net(g)
