from model import make_model


def test_make_model():
    res = make_model()
    assert res == 'da model'
