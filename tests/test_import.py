import golem


def test_name():
    try:
        assert golem.__name__ == 'golem'
    except Exception as e:
        raise e