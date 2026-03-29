"""EmailTool — SMTP delivery with Mailhog in development, real SMTP in production.

Development mode (ENV_MODE=development):
    Emails are captured in an in-memory store (_DEV_INBOX) — no external messages
    are sent. This ensures reproducibility and zero network side-effects during CI
    and automated evaluation.

Production mode (ENV_MODE=production):
    Credentials are injected via SMTP_HOST / SMTP_PORT / SMTP_USER / SMTP_PASS.
    Falls back to Mailhog defaults (localhost:1025) when credentials are absent.
"""

import os
import time
import uuid
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

from tools.base import BaseTool, ToolResult

# ---------------------------------------------------------------------------
# In-memory development inbox — shared across the process lifetime.
# ---------------------------------------------------------------------------
_DEV_INBOX: list[dict] = []


def get_dev_inbox() -> list[dict]:
    """Return a copy of all captured dev-mode emails."""
    return list(_DEV_INBOX)


def clear_dev_inbox() -> None:
    """Clear the development inbox (called on env.reset())."""
    _DEV_INBOX.clear()


class EmailTool(BaseTool):
    name = "email"
    description = (
        "Send a formatted email message to a specified recipient. "
        "In development mode the message is captured in-memory (no SMTP). "
        "In production mode real SMTP credentials are used."
    )
    input_schema = {
        "to": "str — recipient email address",
        "subject": "str — email subject line",
        "body": "str — plain-text or Markdown email body",
        "from_addr": "str — sender address (optional, defaults to env default)",
    }
    output_schema = {
        "sent": "bool — True if delivery was successful",
        "message_id": "str — unique message identifier",
        "recipient": "str",
    }
    cost: float = 0.10

    def run(self, params: dict) -> ToolResult:
        t0 = time.time()
        to_addr: str = str(params.get("to", ""))
        subject: str = str(params.get("subject", "(no subject)"))
        body: str = str(params.get("body", ""))
        from_addr: str = str(
            params.get("from_addr", os.environ.get("SMTP_FROM", "opencesium@example.com"))
        )

        if not to_addr:
            return ToolResult(
                success=False,
                output={"sent": False},
                error="'to' parameter is required",
                latency_ms=(time.time() - t0) * 1000,
            )

        message_id = str(uuid.uuid4())
        env_mode = os.environ.get("ENV_MODE", "development").lower()

        if env_mode != "production":
            # Development mode: capture in memory
            record = {
                "message_id": message_id,
                "to": to_addr,
                "from": from_addr,
                "subject": subject,
                "body": body,
            }
            _DEV_INBOX.append(record)
            return ToolResult(
                success=True,
                output={"sent": True, "message_id": message_id, "recipient": to_addr},
                latency_ms=(time.time() - t0) * 1000,
            )

        # Production mode: real SMTP
        smtp_host = os.environ.get("SMTP_HOST", "localhost")
        smtp_port = int(os.environ.get("SMTP_PORT", "1025"))
        smtp_user = os.environ.get("SMTP_USER", "")
        smtp_pass = os.environ.get("SMTP_PASS", "")

        try:
            msg = MIMEMultipart("alternative")
            msg["Subject"] = subject
            msg["From"] = from_addr
            msg["To"] = to_addr
            msg["Message-ID"] = f"<{message_id}>"
            msg.attach(MIMEText(body, "plain"))

            with smtplib.SMTP(smtp_host, smtp_port, timeout=10) as server:
                if smtp_user and smtp_pass:
                    server.login(smtp_user, smtp_pass)
                server.sendmail(from_addr, [to_addr], msg.as_string())

            return ToolResult(
                success=True,
                output={"sent": True, "message_id": message_id, "recipient": to_addr},
                latency_ms=(time.time() - t0) * 1000,
            )

        except Exception as exc:
            return ToolResult(
                success=False,
                output={"sent": False, "message_id": message_id, "recipient": to_addr},
                error=str(exc),
                latency_ms=(time.time() - t0) * 1000,
            )
