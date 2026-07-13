"""Smoke test: the package imports cleanly and pytest is wired up."""


def test_package_imports():
    import djtagger

    assert djtagger.__version__
