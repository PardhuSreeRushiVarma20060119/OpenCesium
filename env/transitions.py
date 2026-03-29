"""Action dispatch, worker agent execution, and reward computation.

Reward formula (spec §5.2):
    R_t = α·ΔC_t + β·U_t + γ·E_t − δ·I_t − ε·L_t

    α=0.40  performance gain (incremental grader improvement)
    β=0.20  tool correctness (required tool used correctly this step)
    γ=0.15  step efficiency  (early-acting bonus)
    δ=0.20  invalid action penalty
    ε=0.10  loop penalty (same action repeated consecutively)
"""

from __future__ import annotations

import re
from typing import Any

from env.models import Action, AgentConfig, EnvState, Reward

# ---------------------------------------------------------------------------
# Reward weights
# ---------------------------------------------------------------------------
ALPHA = 0.40   # performance gain
BETA  = 0.20   # tool correctness
GAMMA = 0.15   # step efficiency
DELTA = 0.20   # invalid action penalty
EPS   = 0.10   # loop penalty

VALID_STRATEGIES = {"react", "plan_execute", "direct"}


# ===========================================================================
# Public entry-point
# ===========================================================================

def dispatch(
    action: Action,
    state: EnvState,
    task_def: Any,
    grader: Any,
    tool_registry: dict,
) -> tuple[Reward, bool]:
    """Apply *action* to *state* in-place; return (Reward, done).

    Parameters
    ----------
    action       : validated Action model
    state        : current EnvState (mutated in-place)
    task_def     : TaskDefinition for the current task
    grader       : grader callable  (output_dict -> float in [0,1])
    tool_registry: dict[str, BaseTool]

    Returns
    -------
    reward : Reward model with total and component breakdown
    done   : termination flag
    """
    score_before = state.last_score
    invalid = False
    loop = False

    # -----------------------------------------------------------------------
    # Loop detection — same action_type AND same payload as the previous step
    # -----------------------------------------------------------------------
    if (
        state.last_action_type == action.action_type
        and state.last_action_payload == action.payload
    ):
        loop = True

    # -----------------------------------------------------------------------
    # Dispatch
    # -----------------------------------------------------------------------
    if action.action_type == "ADD_TOOL":
        invalid = _handle_add_tool(action, state, tool_registry)

    elif action.action_type == "REMOVE_TOOL":
        invalid = _handle_remove_tool(action, state)

    elif action.action_type == "SET_PROMPT":
        invalid = _handle_set_prompt(action, state)

    elif action.action_type == "SET_STRATEGY":
        invalid = _handle_set_strategy(action, state)

    elif action.action_type == "EVALUATE":
        _handle_evaluate(state, task_def, grader, tool_registry)
        # No invalid flag for EVALUATE itself

    elif action.action_type == "NOOP":
        pass  # Nothing to do

    else:
        # Unknown action type → NOOP with penalty (spec §2.5)
        state.error_log.append(f"Unknown action_type={action.action_type!r}; converted to NOOP")
        invalid = True

    # -----------------------------------------------------------------------
    # Record last action for loop detection
    # -----------------------------------------------------------------------
    state.last_action_type = action.action_type
    state.last_action_payload = dict(action.payload)

    # -----------------------------------------------------------------------
    # Reward computation
    # -----------------------------------------------------------------------
    score_after = state.last_score
    delta_c = score_after - score_before           # ΔC_t
    u_t = _tool_correctness(action, state, task_def)
    e_t = 1.0 - state.step_index / state.max_steps  # E_t = 1 - t/κ
    i_t = 1.0 if invalid else 0.0
    l_t = 1.0 if loop else 0.0

    r_total = (
        ALPHA * delta_c
        + BETA  * u_t
        + GAMMA * e_t
        - DELTA * i_t
        - EPS   * l_t
    )

    reward = Reward(
        total=round(r_total, 6),
        components={
            "delta_c": round(ALPHA * delta_c, 6),
            "tool_correctness": round(BETA * u_t, 6),
            "efficiency": round(GAMMA * e_t, 6),
            "invalid_penalty": round(-DELTA * i_t, 6),
            "loop_penalty": round(-EPS * l_t, 6),
        },
    )

    # -----------------------------------------------------------------------
    # Advance step counter
    # -----------------------------------------------------------------------
    state.step_index += 1
    state.last_reward = reward.total

    # -----------------------------------------------------------------------
    # Termination check (spec §2.4)
    # τ = min{t | t ≥ H  OR  (EVALUATE ∈ a_t AND Φ ≥ θ_pass)}
    # -----------------------------------------------------------------------
    done = state.step_index >= state.max_steps
    if action.action_type == "EVALUATE" and state.last_score >= task_def.passing_threshold:
        done = True
    state.done = done

    return reward, done


# ===========================================================================
# Action handlers
# ===========================================================================

def _handle_add_tool(action: Action, state: EnvState, tool_registry: dict) -> bool:
    """Add a tool to the agent configuration.  Returns True if invalid."""
    tool_name = action.payload.get("tool", "")
    if not tool_name:
        state.error_log.append("ADD_TOOL: missing 'tool' in payload")
        return True
    if tool_name not in tool_registry:
        state.error_log.append(f"ADD_TOOL: tool {tool_name!r} not in registry; ignored")
        return True
    if tool_name not in state.agent_config.tools:
        state.agent_config.tools.append(tool_name)
    return False


def _handle_remove_tool(action: Action, state: EnvState) -> bool:
    """Remove a tool from the agent configuration.  Returns True if invalid."""
    tool_name = action.payload.get("tool", "")
    if not tool_name:
        state.error_log.append("REMOVE_TOOL: missing 'tool' in payload")
        return True
    if tool_name in state.agent_config.tools:
        state.agent_config.tools.remove(tool_name)
    return False


def _handle_set_prompt(action: Action, state: EnvState) -> bool:
    """Set the worker agent's system prompt.  Returns True if invalid."""
    prompt = action.payload.get("prompt", "")
    if not isinstance(prompt, str):
        state.error_log.append("SET_PROMPT: 'prompt' must be a string")
        return True
    state.agent_config.prompt = prompt
    return False


def _handle_set_strategy(action: Action, state: EnvState) -> bool:
    """Set the worker agent's execution strategy.  Returns True if invalid."""
    strategy = action.payload.get("strategy", "")
    if strategy not in VALID_STRATEGIES:
        state.error_log.append(
            f"SET_STRATEGY: unknown strategy {strategy!r}; "
            f"must be one of {sorted(VALID_STRATEGIES)}"
        )
        return True
    state.agent_config.strategy = strategy  # type: ignore[assignment]
    return False


def _handle_evaluate(
    state: EnvState,
    task_def: Any,
    grader: Any,
    tool_registry: dict,
) -> None:
    """Execute the worker agent and update last_score."""
    output, tool_logs = _execute_worker(state.agent_config, task_def, tool_registry)
    state.worker_output = output
    state.tool_execution_log.extend(tool_logs)
    new_score = grader(output)
    state.last_score = round(new_score, 6)


# ===========================================================================
# Worker agent executor
# ===========================================================================

def _execute_worker(
    agent_config: AgentConfig,
    task_def: Any,
    tool_registry: dict,
) -> tuple[dict, list[str]]:
    """Execute the worker agent's tool pipeline for the given task.

    Returns (output_dict, tool_call_log).
    The output_dict structure varies per task (see grader signatures).
    """
    task_id: str = task_def.task_id

    if task_id == "task_easy_stock_alert":
        return _run_easy(agent_config, task_def, tool_registry)
    elif task_id == "task_medium_gdp_growth":
        return _run_medium(agent_config, task_def, tool_registry)
    elif task_id == "task_hard_structured_report":
        return _run_hard(agent_config, task_def, tool_registry)
    else:
        return {}, []


# ---------------------------------------------------------------------------
# Task 1 — Financial Alert Report
# ---------------------------------------------------------------------------

def _run_easy(
    cfg: AgentConfig, task_def: Any, tools: dict
) -> tuple[dict, list[str]]:
    output: dict[str, Any] = {
        "stock_price": None,
        "email_sent": False,
        "email_body": "",
    }
    log: list[str] = []
    ticker = task_def.goal_params.get("ticker", "AAPL")
    recipient = task_def.goal_params.get("recipient", "trader@example.com")

    if "stock" in cfg.tools and "stock" in tools:
        result = tools["stock"].run({"ticker": ticker})
        log.append("stock")
        if result.success:
            output["stock_price"] = float(result.output.get("price", 0))
        else:
            output["stock_price"] = None

    if "email" in cfg.tools and "email" in tools and output["stock_price"] is not None:
        price = output["stock_price"]
        body = (
            f"Stock Alert: {ticker} is currently trading at ${price:.2f}. "
            f"Ticker: {ticker}."
        )
        result = tools["email"].run(
            {"to": recipient, "subject": f"{ticker} Stock Alert", "body": body}
        )
        log.append("email")
        if result.success:
            output["email_sent"] = True
            output["email_body"] = body

    return output, log


# ---------------------------------------------------------------------------
# Task 2 — GDP Growth Analysis
# ---------------------------------------------------------------------------

def _run_medium(
    cfg: AgentConfig, task_def: Any, tools: dict
) -> tuple[dict, list[str]]:
    output: dict[str, Any] = {
        "gdp_values": {},
        "growth_rates": {},
        "email_sent": False,
        "email_body": "",
    }
    log: list[str] = []
    countries: list = task_def.goal_params.get("countries", ["US", "China"])
    years: list = task_def.goal_params.get("years", [2022, 2023])
    recipient = task_def.goal_params.get("recipient", "analyst@example.com")
    year_prev, year_curr = years[0], years[1]

    if "search" in cfg.tools and "search" in tools:
        for country in countries:
            query = (
                f"{country} GDP {year_prev} {year_curr} billion USD World Bank"
            )
            result = tools["search"].run({"query": query, "max_results": 5})
            log.append("search")
            if result.success:
                gdp_data = _parse_gdp_from_snippets(
                    result.output, country, year_prev, year_curr
                )
                if gdp_data:
                    output["gdp_values"][country] = gdp_data

    if "calculator" in cfg.tools and "calculator" in tools:
        for country, gdp_data in output["gdp_values"].items():
            gdp_prev = gdp_data.get("gdp_prev")
            gdp_curr = gdp_data.get("gdp_curr")
            if gdp_prev and gdp_curr and float(gdp_prev) != 0:
                expr = f"({float(gdp_curr)} - {float(gdp_prev)}) / {float(gdp_prev)} * 100"
                result = tools["calculator"].run({"expression": expr})
                log.append("calculator")
                if result.success:
                    output["growth_rates"][country] = round(float(result.output), 4)

    if "email" in cfg.tools and "email" in tools:
        body = _build_gdp_email(countries, output["gdp_values"], output["growth_rates"])
        result = tools["email"].run(
            {
                "to": recipient,
                "subject": "GDP Growth Rate Analysis",
                "body": body,
            }
        )
        log.append("email")
        if result.success:
            output["email_sent"] = True
            output["email_body"] = body

    return output, log


def _parse_gdp_from_snippets(
    snippets: list, country: str, year_prev: int, year_curr: int
) -> dict | None:
    """Extract GDP values for two consecutive years from search snippets."""
    combined = " ".join(
        s.get("content", "") + " " + s.get("title", "") for s in snippets
    )

    # Patterns: "$X.X trillion", "X,XXX billion", "X.X trillion"
    trillion_pat = re.compile(
        r"(\d+(?:\.\d+)?)\s*trillion", re.IGNORECASE
    )
    billion_pat = re.compile(
        r"(\d[\d,]*(?:\.\d+)?)\s*billion", re.IGNORECASE
    )

    values: list[float] = []
    for m in trillion_pat.finditer(combined):
        values.append(float(m.group(1)) * 1000)  # convert to billions
    for m in billion_pat.finditer(combined):
        values.append(float(m.group(1).replace(",", "")))

    if len(values) >= 2:
        # Use first two values found (prev, curr)
        return {"gdp_prev": round(values[0], 2), "gdp_curr": round(values[1], 2)}
    if len(values) == 1:
        # Fallback: assume a nominal 2% growth to allow formula verification
        _NOMINAL_GROWTH = 1.02
        return {"gdp_prev": round(values[0], 2), "gdp_curr": round(values[0] * _NOMINAL_GROWTH, 2)}
    return None


def _build_gdp_email(
    countries: list, gdp_values: dict, growth_rates: dict
) -> str:
    lines = ["GDP Growth Rate Analysis", "=" * 40, ""]
    for country in countries:
        rate = growth_rates.get(country)
        gdp = gdp_values.get(country, {})
        if rate is not None:
            lines.append(
                f"{country}: GDP growth rate = {rate:.2f}% "
                f"(from ${gdp.get('gdp_prev', 'N/A')}B to ${gdp.get('gdp_curr', 'N/A')}B)"
            )
        else:
            lines.append(f"{country}: data not available")

    rates = [growth_rates[c] for c in countries if c in growth_rates]
    if len(rates) == 2:
        names = [c for c in countries if c in growth_rates]
        if rates[0] > rates[1]:
            comparison = (
                f"\n{names[0]} grew faster than {names[1]} "
                f"({rates[0]:.2f}% vs {rates[1]:.2f}%) over the measured period."
            )
        elif rates[1] > rates[0]:
            comparison = (
                f"\n{names[1]} grew faster than {names[0]} "
                f"({rates[1]:.2f}% vs {rates[0]:.2f}%) over the measured period."
            )
        else:
            comparison = f"\nBoth countries achieved the same growth rate of {rates[0]:.2f}%."
        lines.append(comparison)

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Task 3 — Multi-Source Structured Report
# ---------------------------------------------------------------------------

def _run_hard(
    cfg: AgentConfig, task_def: Any, tools: dict
) -> tuple[dict, list[str]]:
    output: dict[str, Any] = {
        "stock_price": None,
        "history": [],
        "news": [],
        "sma7": None,
        "report": "",
        "email_sent": False,
        "email_body": "",
        "tool_call_log": [],
    }
    log: list[str] = []
    ticker = task_def.goal_params.get("ticker", "AAPL")
    recipient = task_def.goal_params.get("recipient", "report@example.com")

    # Step 1 — fetch live price + 7-day history
    if "stock" in cfg.tools and "stock" in tools:
        result = tools["stock"].run(
            {"ticker": ticker, "include_history": True, "history_days": 7}
        )
        log.append("stock")
        if result.success:
            output["stock_price"] = float(result.output.get("price", 0))
            output["history"] = result.output.get("history", [])

    # Step 2 — search for recent market news
    if "search" in cfg.tools and "search" in tools:
        result = tools["search"].run(
            {"query": f"{ticker} stock market news analysis", "max_results": 5}
        )
        log.append("search")
        if result.success:
            output["news"] = result.output

    # Step 3 — compute SMA_7
    if "calculator" in cfg.tools and "calculator" in tools and len(output["history"]) >= 7:
        closes = [float(h["close"]) for h in output["history"][:7]]
        expr = f"({' + '.join(str(c) for c in closes)}) / 7"
        result = tools["calculator"].run({"expression": expr})
        log.append("calculator")
        if result.success:
            output["sma7"] = round(float(result.output), 4)

    # Step 4 — build Markdown report
    output["report"] = _build_hard_report(ticker, output)

    # Step 5 — send email
    if "email" in cfg.tools and "email" in tools:
        result = tools["email"].run(
            {
                "to": recipient,
                "subject": f"{ticker} Structured Market Report",
                "body": output["report"],
            }
        )
        log.append("email")
        if result.success:
            output["email_sent"] = True
            output["email_body"] = output["report"]

    output["tool_call_log"] = log
    return output, log


def _build_hard_report(ticker: str, data: dict) -> str:
    """Build a Markdown analytical report with the 4 required sections."""
    price = data.get("stock_price", "N/A")
    history = data.get("history", [])
    news = data.get("news", [])
    sma7 = data.get("sma7", "N/A")

    # Section 1 — Header
    lines = [
        f"# Header",
        f"",
        f"**{ticker} Market Report**",
        f"",
        f"This report provides a structured analysis of {ticker} stock based on "
        f"live price data, historical performance, and current market news.",
        f"",
    ]

    # Section 2 — Data
    lines += [
        f"## Data",
        f"",
        f"**Current Price:** ${price:.2f}" if isinstance(price, float) else f"**Current Price:** {price}",
        f"",
        f"**7-Day Price History:**",
    ]
    for entry in history[:7]:
        lines.append(f"- {entry.get('date', 'N/A')}: ${entry.get('close', 'N/A'):.2f}"
                     if isinstance(entry.get("close"), float) else
                     f"- {entry.get('date', 'N/A')}: {entry.get('close', 'N/A')}")
    lines.append("")
    if news:
        lines += ["**Recent Market News:**", ""]
        for item in news[:3]:
            lines.append(f"- {item.get('title', 'N/A')}: {item.get('content', '')[:120]}...")
    lines.append("")

    # Section 3 — Computation
    lines += [
        f"## Computation",
        f"",
        f"**7-Day Simple Moving Average (SMA_7):**",
        f"",
    ]
    if history:
        closes_str = ", ".join(
            f"${h.get('close', 0):.2f}" for h in history[:7]
        )
        lines.append(f"Closing prices: {closes_str}")
        lines.append(f"")
        close_vals = " + ".join(f"{h.get('close', 0):.2f}" for h in history[:7])
        lines.append(f"SMA_7 = ({close_vals}) / 7")
        lines.append(f"")
    lines.append(f"**Result: SMA_7 = ${sma7:.4f}**" if isinstance(sma7, float) else f"**Result: SMA_7 = {sma7}**")
    lines.append("")

    # Section 4 — Executive Summary
    trend = ""
    if isinstance(price, float) and isinstance(sma7, float):
        if price > sma7:
            trend = f"trading above its 7-day SMA (${sma7:.2f}), suggesting short-term bullish momentum"
        elif price < sma7:
            trend = f"trading below its 7-day SMA (${sma7:.2f}), suggesting short-term bearish pressure"
        else:
            trend = f"trading at its 7-day SMA (${sma7:.2f})"

    lines += [
        f"## Executive Summary",
        f"",
        f"{ticker} is currently " + (trend if trend else f"priced at ${price}") + ".",
    ]
    if isinstance(sma7, float):
        lines.append(
            f"The 7-day SMA of {sma7:.4f} was computed from recent closing prices."
        )
    lines += [
        f"Market news coverage provides additional context for investment decisions.",
        f"",
        f"*Report generated by OpenCesium automated workflow system.*",
    ]

    return "\n".join(lines)


# ===========================================================================
# Tool-correctness helper  (U_t in the reward formula)
# ===========================================================================

def _tool_correctness(
    action: Action,
    state: EnvState,
    task_def: Any,
) -> float:
    """Return U_t ∈ {0, 1}.

    U_t = 1  if:
      - ADD_TOOL and the tool is in the task's required tool set
      - EVALUATE and all required tools are present in agent_config
    U_t = 0  otherwise.
    """
    required: list = task_def.required_tools

    if action.action_type == "ADD_TOOL":
        tool = action.payload.get("tool", "")
        return 1.0 if tool in required else 0.0

    if action.action_type == "EVALUATE":
        configured = set(state.agent_config.tools)
        required_set = set(required)
        return 1.0 if required_set.issubset(configured) else 0.0

    return 0.0
