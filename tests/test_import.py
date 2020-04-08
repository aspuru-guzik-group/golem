import golem


def test_name():
    try:
        assert golem.__name__ == 'golem'
        version = golem.__version__
    except Exception as e:
        raise e