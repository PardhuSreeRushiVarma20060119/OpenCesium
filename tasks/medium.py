"""Task 2 — GDP Growth Analysis (Medium).

Required tools : search, calculator, email
Max steps      : 15
Passing thresh : Φ ≥ 0.7

Formula        : g = (GDP_t - GDP_{t-1}) / GDP_{t-1} × 100 %

Grader components:
  f1  GDP data retrieved for both countries        w=0.25
  f2  Growth rate computed correctly (±0.5 %)      w=0.30
  f3  Email sent                                   w=0.25
  f4  Report contains percentage + comparison      w=0.20
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


TASK_MEDIUM = TaskDefinition(
    task_id="task_medium_gdp_growth",
    description=(
        "Search the web for GDP values of US and China for years 2022 and 2023. "
        "Compute the year-over-year growth rate for each country using the formula "
        "g = (GDP_t - GDP_{t-1}) / GDP_{t-1} × 100%. "
        "Deliver a structured email summary including country names, numeric growth "
        "rates, and a one-sentence comparative statement."
    ),
    required_tools=["search", "calculator", "email"],
    goal_params={
        "countries": ["US", "China"],
        "years": [2022, 2023],
        "recipient": "analyst@example.com",
    },
    max_steps=15,
    passing_threshold=0.7,
    grader_weights={"f1": 0.25, "f2": 0.30, "f3": 0.25, "f4": 0.20},
)


def grade_medium(output: dict) -> float:
    """Deterministic grader for task_medium_gdp_growth.

    Args:
        output: dict produced by the worker executor, containing keys:
            gdp_values  (dict[country -> {gdp_prev, gdp_curr}]),
            growth_rates (dict[country -> float %]),
            email_sent  (bool),
            email_body  (str).

    Returns:
        Score in [0, 1].
    """
    s = 0.0

    gdp_values: dict = output.get("gdp_values", {})
    growth_rates: dict = output.get("growth_rates", {})

    # f1 — GDP data retrieved for both countries (2 numeric values present)
    valid_countries = [
        c
        for c, v in gdp_values.items()
        if isinstance(v.get("gdp_prev"), (int, float))
        and isinstance(v.get("gdp_curr"), (int, float))
    ]
    if len(valid_countries) >= 2:
        s += 0.25

    # f2 — growth rate computed correctly (±0.5 %) for at least one country
    correct_rates = 0
    for country in valid_countries:
        if country not in growth_rates:
            continue
        gdp_prev = float(gdp_values[country]["gdp_prev"])
        gdp_curr = float(gdp_values[country]["gdp_curr"])
        if gdp_prev == 0:
            continue
        expected = (gdp_curr - gdp_prev) / gdp_prev * 100.0
        if abs(float(growth_rates[country]) - expected) < 0.5:
            correct_rates += 1
    if correct_rates >= 1:
        s += 0.30

    # f3 — email sent
    if output.get("email_sent") is True:
        s += 0.25

    # f4 — report contains percentage sign and a comparison keyword
    body: str = str(output.get("email_body", ""))
    has_percent = "%" in body
    has_comparison = any(
        kw in body.lower()
        for kw in [
            "compared",
            "higher",
            "lower",
            "versus",
            " vs ",
            "growth",
            "faster",
            "slower",
            "greater",
            "less",
        ]
    )
    if has_percent and has_comparison:
        s += 0.20

    return min(s, 1.0)
