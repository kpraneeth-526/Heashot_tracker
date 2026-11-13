"""Email the headshot to the student over SMTP.

Defaults point at Mailpit (a fake inbox: `mailpit`, then open
http://localhost:8025), so testing can never reach a real inbox.

To send real email, set these (for example in a git-ignored .env file):
    SMTP_HOST, SMTP_PORT        the mail server, e.g. smtp.gmail.com, 587
    SMTP_USER, SMTP_PASSWORD    login; turns on encryption (STARTTLS)
    MAIL_FROM                   sender address (defaults to SMTP_USER)
A real deployment points these at the university's own mail server.
"""

import os
import smtplib
import ssl
from email.message import EmailMessage
from pathlib import Path

SUBJECT = "Your career fair headshot"


def load_env_file(path: str | Path = ".env") -> None:
    """Read KEY=VALUE lines from a .env file into the environment. Settings
    already in the environment (e.g. typed on the command line) win."""
    path = Path(path)
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("'\""))


def build_message(to: str, name: str, photo: bytes, sender: str) -> EmailMessage:
    msg = EmailMessage()
    msg["Subject"] = SUBJECT
    msg["From"] = sender
    msg["To"] = to
    greeting = f"Hi {name}," if name else "Hi,"
    msg.set_content(
        f"{greeting}\n\n"
        "Here is your headshot from the career fair booth, attached to this email.\n\n"
        "The booth deleted its copy as soon as this email was sent.\n"
    )
    msg.add_attachment(photo, maintype="image", subtype="jpeg", filename="headshot.jpg")
    return msg


def send_email(to: str, name: str, photo: bytes) -> None:
    """Send the headshot. Raises on any failure, so the caller can log it."""
    host = os.environ.get("SMTP_HOST", "localhost")
    port = int(os.environ.get("SMTP_PORT", "1025"))
    user = os.environ.get("SMTP_USER", "")
    password = os.environ.get("SMTP_PASSWORD", "")
    sender = os.environ.get("MAIL_FROM") or user or "headshot-booth@localhost"
    with smtplib.SMTP(host, port, timeout=10) as smtp:
        if user:
            smtp.starttls(context=ssl.create_default_context())
            smtp.login(user, password)
        smtp.send_message(build_message(to, name, photo, sender))


def describe_server() -> str:
    """Where email will go, for the startup log (no password)."""
    host = os.environ.get("SMTP_HOST", "localhost")
    port = os.environ.get("SMTP_PORT", "1025")
    if host == "localhost" and port == "1025":
        return "Mailpit (fake inbox, http://localhost:8025)"
    who = os.environ.get("SMTP_USER") or "no login"
    return f"{host}:{port} as {who} (REAL email)"
