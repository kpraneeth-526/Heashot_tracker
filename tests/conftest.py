import pytest


@pytest.fixture(autouse=True)
def no_real_dslr(monkeypatch):
    """Tests never look for a real camera on USB; DSLR tests use a fake one."""
    monkeypatch.setenv("DSLR", "0")
