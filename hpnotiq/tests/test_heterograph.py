import pytest

def test_hypergraph():
    import hpnotiq as hq
    from rdkit import Chem
    g = hq.graph.from_rdkit_mol(
        Chem.MolFromSmiles(
            'c1ccccc1'
        )
    )


    hg = hq.heterograph.from_homograph(
        g
    )
