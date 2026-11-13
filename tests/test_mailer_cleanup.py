import os

import pytest

from app import mailer
from app.cleanup import log_event, purge, read_events
from eval import make_lfw_gallery

MAIL_SETTINGS = ("SMTP_HOST", "SMTP_PORT", "SMTP_USER", "SMTP_PASSWORD", "MAIL_FROM")


@pytest.fixture(autouse=True)
def no_real_mail_settings(monkeypatch):
    """Ignore any real login from the developer's .env during tests."""
    for key in MAIL_SETTINGS:
        monkeypatch.delenv(key, raising=False)


class FakeSMTP:
    sent = []
    events = []

    def __init__(self, host, port, timeout):
        self.where = (host, port)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def starttls(self, context):
        FakeSMTP.events.append("starttls")

    def login(self, user, password):
        FakeSMTP.events.append(("login", user, password))

    def send_message(self, msg):
        FakeSMTP.sent.append((self.where, msg))


def test_send_email_attaches_photo(monkeypatch):
    FakeSMTP.sent, FakeSMTP.events = [], []
    monkeypatch.setattr(mailer.smtplib, "SMTP", FakeSMTP)
    mailer.send_email("ada@example.edu", "Ada", b"\xff\xd8jpeg")

    [(where, msg)] = FakeSMTP.sent
    assert where == ("localhost", 1025)  # Mailpit by default
    assert msg["To"] == "ada@example.edu"
    assert "Hi Ada," in msg.get_body(("plain",)).get_content()
    [attachment] = msg.iter_attachments()
    assert attachment.get_filename() == "headshot.jpg"
    assert attachment.get_content() == b"\xff\xd8jpeg"
    assert FakeSMTP.events == []  # Mailpit: no login, no encryption


def test_send_email_uses_configured_server(monkeypatch):
    FakeSMTP.sent = []
    monkeypatch.setattr(mailer.smtplib, "SMTP", FakeSMTP)
    monkeypatch.setenv("SMTP_HOST", "mail.example.edu")
    monkeypatch.setenv("SMTP_PORT", "25")
    mailer.send_email("ada@example.edu", "", b"x")
    [(where, msg)] = FakeSMTP.sent
    assert where == ("mail.example.edu", 25)
    assert msg.get_body(("plain",)).get_content().startswith("Hi,")


def test_purge_overwrites_photo():
    photo = bytearray(b"secret photo")
    purge(photo)
    assert photo == bytearray()


def test_log_event(tmp_path):
    db = tmp_path / "audit.db"
    log_event(db, "sent")
    log_event(db, "failed")
    assert [s for _, s in read_events(db)] == ["sent", "failed"]
    with pytest.raises(ValueError):
        log_event(db, "ada@example.edu")


def test_read_events_without_log(tmp_path):
    assert read_events(tmp_path / "missing.db") == []


def test_send_email_with_login_encrypts_and_logs_in(monkeypatch):
    FakeSMTP.sent, FakeSMTP.events = [], []
    monkeypatch.setattr(mailer.smtplib, "SMTP", FakeSMTP)
    monkeypatch.setenv("SMTP_HOST", "smtp.gmail.com")
    monkeypatch.setenv("SMTP_PORT", "587")
    monkeypatch.setenv("SMTP_USER", "booth@gmail.com")
    monkeypatch.setenv("SMTP_PASSWORD", "app password")
    mailer.send_email("friend@uga.edu", "Friend", b"x")
    assert FakeSMTP.events == ["starttls", ("login", "booth@gmail.com", "app password")]
    [(where, msg)] = FakeSMTP.sent
    assert where == ("smtp.gmail.com", 587)
    assert msg["From"] == "booth@gmail.com"
    assert "REAL email" in mailer.describe_server()
    assert "app password" not in mailer.describe_server()


def test_describe_server_defaults_to_mailpit():
    assert "Mailpit" in mailer.describe_server()


def test_load_env_file(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text("# comment\nSMTP_USER=booth@gmail.com\nSMTP_PASSWORD='abcd efgh'\n"
                   "SMTP_HOST=from-file\n\nnot a setting\n")
    monkeypatch.setattr(os, "environ", dict(os.environ))  # don't leak into other tests
    os.environ["SMTP_HOST"] = "from-command-line"
    mailer.load_env_file(env)
    assert os.environ["SMTP_USER"] == "booth@gmail.com"
    assert os.environ["SMTP_PASSWORD"] == "abcd efgh"
    assert os.environ["SMTP_HOST"] == "from-command-line"  # command line wins
    mailer.load_env_file(tmp_path / "missing.env")  # no file: nothing happens


def test_lfw_gallery_plus_addresses(tmp_path):
    lfw = tmp_path / "lfw"
    for person in ("Serena_Williams", "Dennis_O'Neil"):
        (lfw / person).mkdir(parents=True)
        (lfw / person / f"{person}_0001.jpg").write_bytes(b"x")
    out = tmp_path / "gallery.csv"
    make_lfw_gallery.main(["--lfw", str(lfw), "--out", str(out), "--people", "0",
                           "--email-to", "me@gmail.com"])
    rows = out.read_text().splitlines()[1:]
    assert any("me+serena_williams@gmail.com" in r for r in rows)
    assert any("me+dennis_oneil@gmail.com" in r for r in rows)
