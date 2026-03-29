"""Task 1 — Financial Alert Report (Easy).

Required tools : stock, email
Max steps      : 10
Passing thresh : Φ ≥ 0.7

Grader components:
  f1  Stock price retrieved and numeric          w=0.40
  f2  Email sent (receipt confirmed)              w=0.40
  f3  Ticker symbol present in email body         w=0.20
"""

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


TASK_EASY = TaskDefinition(
    task_id="task_easy_stock_alert",
    description=(
        "Retrieve the real-time price of AAPL from a financial data API, "
        "format a concise one-sentence alert, and deliver it to a specified "
        "recipient via email."
    ),
    required_tools=["stock", "email"],
    goal_params={
        "ticker": "AAPL",
        "recipient": "trader@example.com",
    },
    max_steps=10,
    passing_threshold=0.7,
    grader_weights={"f1": 0.40, "f2": 0.40, "f3": 0.20},
)


def grade_easy(output: dict) -> float:
    """Deterministic grader for task_easy_stock_alert.

    Args:
        output: dict produced by the worker executor, containing keys:
            stock_price (float|None), email_sent (bool), email_body (str).

    Returns:
        Score in [0, 1].
    """
    s = 0.0

    # f1 — stock price retrieved and numeric
    if isinstance(output.get("stock_price"), (int, float)) and output.get("stock_price") is not None:
        s += 0.40

    # f2 — email sent (receipt confirmed)
    if output.get("email_sent") is True:
        s += 0.40

    # f3 — ticker symbol present in email body
    if "AAPL" in str(output.get("email_body", "")):
        s += 0.20

    return min(s, 1.0)
