import re
import html

URL_RE = re.compile(r"https?://\S+|www\.\S+")
EMAIL_RE = re.compile(r"\S+@\S+\.\S+")
TAG_RE = re.compile(r"<[^>]+>")
STYLE_RE = re.compile(r"<(script|style).*?>.*?</\1>", re.S | re.I)
SUBJECT_RE = re.compile(r"^subject:\s*(.*)$", re.I | re.M)


def strip_html(text: str) -> str:
    text = STYLE_RE.sub(" ", text)
    text = TAG_RE.sub(" ", text)
    return html.unescape(text)


def prepare_email(raw: str) -> str:
    """Take a raw email (optionally starting with 'Subject: ...') and return cleaned text."""
    raw = str(raw or "")

    subject = ""
    match = SUBJECT_RE.search(raw[:1000])
    if match:
        subject = match.group(1)
        raw = raw[:match.start()] + raw[match.end():]

    text = f"{subject} {subject} {raw}"
    text = strip_html(text).lower()
    text = URL_RE.sub(" urltoken ", text)
    text = EMAIL_RE.sub(" emailtoken ", text)
    text = re.sub(r"\d+", " numtoken ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text
