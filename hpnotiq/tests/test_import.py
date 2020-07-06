import pytest

def test_import():
    import hpnotiq as hq

def test_data():
    import hpnotiq as hq
    assert len(hq.data.esol()[:10]) == 10
