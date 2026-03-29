"""Task 3 — Multi-Source Structured Report (Hard).

Required tools : stock, search, calculator, email
Max steps      : 20
Passing thresh : Φ ≥ 0.7

SMA formula    : SMA_7 = (1/7) * Σ_{i=0}^{6} p_{t-i}

Grader components:
  f1  Stock price + 7-day history retrieved         w=0.15
  f2  News snippets fetched                         w=0.15
  f3  SMA_7 correct (±0.01)                         w=0.20
  f4  Report contains ≥4 required sections          w=0.20
  f5  Email sent with full report body              w=0.20
  f6  No redundant consecutive tool calls           w=0.10

Required report sections (case-insensitive heading match):
  Header | Data | Computation | Executive Summary
"""

import re
from dataclasses import dataclass, field


@dataclass
class TaskDefinition:
    task_id: str
    description: str
    required_tools: list
    goal_params: dict
    max_steps: int
    passing_threshold: float
    grader_weights: dict = field(default_factory=dict)


_REQUIRED_SECTIONS = ["header", "data", "computation", "executive summary"]


TASK_HARD = TaskDefinition(
    task_id="task_hard_structured_report",
    description=(
        "Build a structured Markdown analytical report by: "
        "(1) fetching live AAPL stock data and 7-day price history; "
        "(2) searching for recent market news about AAPL; "
        "(3) computing a 7-day simple moving average (SMA_7); "
        "(4) generating a report with at least four named sections "
        "(Header, Data, Computation, Executive Summary); "
        "(5) delivering the full report via email."
    ),
    required_tools=["stock", "search", "calculator", "email"],
    goal_params={
        "ticker": "AAPL",
        "recipient": "report@example.com",
        "history_days": 7,
    },
    max_steps=20,
    passing_threshold=0.7,
    grader_weights={"f1": 0.15, "f2": 0.15, "f3": 0.20, "f4": 0.20, "f5": 0.20, "f6": 0.10},
)


def _count_valid_sections(report: str) -> int:
    """Count how many required sections appear as headings in the report."""
    report_lower = report.lower()
    count = 0
    for section in _REQUIRED_SECTIONS:
        # Match Markdown headings: # Section, ## Section, or plain "Section" line
        pattern = rf"(?:^|\n)#{{0,3}}\s*{re.escape(section)}"
        if re.search(pattern, report_lower):
            count += 1
    return count


def grade_hard(output: dict) -> float:
    """Deterministic grader for task_hard_structured_report.

    Args:
        output: dict produced by the worker executor, containing keys:
            stock_price (float|None),
            history     (list[dict] with ≥7 entries, each having 'close'),
            news        (list[dict] — search results),
            sma7        (float|None — computed SMA_7),
            report      (str — Markdown report),
            email_sent  (bool),
            email_body  (str),
            tool_call_log (list[str] — ordered tool invocation log).

    Returns:
        Score in [0, 1].
    """
    s = 0.0

    # f1 — stock price + 7-day history (need price + at least 7 close values)
    history: list = output.get("history", [])
    has_price = isinstance(output.get("stock_price"), (int, float))
    has_history = len(history) >= 7 and all(
        isinstance(entry.get("close"), (int, float)) for entry in history[:7]
    )
    if has_price and has_history:
        s += 0.15

    # f2 — news snippets fetched (≥1 result)
    news: list = output.get("news", [])
    if len(news) >= 1:
        s += 0.15

    # f3 — SMA_7 computed correctly (±0.01)
    sma7 = output.get("sma7")
    if has_history and sma7 is not None:
        closes = [float(h["close"]) for h in history[:7]]
        expected_sma = sum(closes) / 7.0
        if abs(float(sma7) - expected_sma) < 0.01:
            s += 0.20

    # f4 — report contains ≥4 required sections
    report: str = str(output.get("report", ""))
    if _count_valid_sections(report) >= 4:
        s += 0.20

    # f5 — email sent with full report body (body length > 200 chars)
    if output.get("email_sent") is True and len(str(output.get("email_body", ""))) > 200:
        s += 0.20

    # f6 — no redundant consecutive tool calls
    tool_log: list = output.get("tool_call_log", [])
    has_redundancy = any(
        tool_log[i] == tool_log[i - 1] for i in range(1, len(tool_log))
    )
    if not has_redundancy:
        s += 0.10

    return min(s, 1.0)
