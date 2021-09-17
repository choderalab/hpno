import pytest

def test_import():
    import hpno
    from hpno import heterograph
    from hpno import heterograph
    from hpno.heterograph import heterograph

def test_methane():
    import hpno
    import dgllife
    g = dgllife.utils.smiles_to_bigraph("C", explicit_hydrogens=True)
    g = hpno.heterograph(g)

    print(g.nodes['n2'].data["idxs"])
    assert g.number_of_nodes("n1") == 5
    assert g.number_of_nodes("n2") == 8
    assert g.number_of_nodes("n3") == 12
    assert g.number_of_nodes("n4") == 0
