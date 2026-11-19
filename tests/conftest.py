import pytest


@pytest.fixture(autouse=True)
def no_real_dslr(monkeypatch):
    """Tests never look for a real camera on USB; DSLR tests use a fake one."""
    monkeypatch.setenv("DSLR", "0")


@pytest.fixture(autouse=True)
def send_while_waiting(monkeypatch):
    """Most tests check the email right after confirming, so they send it
    before replying. Background sending has its own tests."""
    monkeypatch.setenv("SEND_IN_BACKGROUND", "0")
