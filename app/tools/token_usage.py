"""Counts the tokens each agent sends to its LLM, so we can see where the
prompt budget actually goes.

WHY THIS EXISTS
---------------
Agents 4 and 6 do not make one call each — they call once per ontology, so a
single /query fans out into many requests. The console timings show which STEP
is slow, but not which step is EXPENSIVE. A cheap agent called nine times can
easily out-spend the one big Agent 7 prompt, and there was no way to see that.

HOW IT WORKS
------------
The provider clients (ollama, gemini, openai, anthropic) know the token counts
but not which agent they are serving. The router knows the agent but not the
counts. So the router publishes the agent name into a ContextVar for the
duration of the call, and the client reads it back when it records usage.

A ContextVar rather than a global: agents 4, 6 and 7 run their per-ontology
calls concurrently with asyncio.gather, and gather wraps each coroutine in its
own Task with its own copy of the context. A plain global would be overwritten
by whichever call started last.

USAGE
-----
    reset()                       # start of a request
    ... run the pipeline ...
    print(format_report())        # human-readable table
    snapshot()                    # same numbers as a dict, for the API response
"""
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
import threading

# Set by the router around each call; read by the provider clients.
_current_agent: ContextVar = ContextVar("current_agent", default=None)

_lock = threading.Lock()


@dataclass
class AgentUsage:
    """Running totals for one agent across a request."""
    agent: str
    provider: str = ""
    model: str = ""
    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    # Biggest single prompt seen. A large total can mean one huge prompt or many
    # small ones, and the fix is different in each case.
    max_prompt_tokens: int = 0
    # Calls whose token counts the provider did not report (see record()).
    calls_without_usage: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


_usage: dict = {}


@contextmanager
def agent_scope(agent_key: str):
    """Mark every LLM call inside this block as belonging to `agent_key`."""
    token = _current_agent.set(agent_key)
    try:
        yield
    finally:
        _current_agent.reset(token)


def record(provider: str, model: str, usage) -> None:
    """Add one call's token counts to the current agent's totals.

    `usage` is the provider's usage object. Never raises: token accounting is
    diagnostics, and a missing field must not take down a working pipeline.
    """
    agent = _current_agent.get() or "(outside an agent)"

    prompt = _read(usage, "prompt_tokens", "input_tokens")
    completion = _read(usage, "completion_tokens", "output_tokens")

    with _lock:
        entry = _usage.get(agent)
        if entry is None:
            entry = _usage[agent] = AgentUsage(agent=agent)
        entry.provider = provider
        entry.model = model
        entry.calls += 1
        if prompt is None and completion is None:
            # Some providers omit usage entirely (streaming, or the HF
            # inference API). Counting the call but not the tokens keeps the
            # report honest instead of silently under-reporting.
            entry.calls_without_usage += 1
            return
        entry.prompt_tokens += prompt or 0
        entry.completion_tokens += completion or 0
        entry.max_prompt_tokens = max(entry.max_prompt_tokens, prompt or 0)


def _read(usage, *names):
    """First attribute/key present on the usage object, or None."""
    if usage is None:
        return None
    for name in names:
        value = getattr(usage, name, None)
        if value is None and isinstance(usage, dict):
            value = usage.get(name)
        if isinstance(value, int):
            return value
    return None


def reset() -> None:
    """Clear all counters. Call once at the start of a request."""
    with _lock:
        _usage.clear()


def snapshot() -> dict:
    """Current totals as plain data, biggest prompt spender first."""
    with _lock:
        entries = sorted(_usage.values(), key=lambda e: e.prompt_tokens, reverse=True)
        rows = [
            {
                "agent": e.agent,
                "provider": e.provider,
                "model": e.model,
                "calls": e.calls,
                "prompt_tokens": e.prompt_tokens,
                "completion_tokens": e.completion_tokens,
                "total_tokens": e.total_tokens,
                "max_prompt_tokens": e.max_prompt_tokens,
                "avg_prompt_tokens": round(e.prompt_tokens / e.calls) if e.calls else 0,
                "calls_without_usage": e.calls_without_usage,
            }
            for e in entries
        ]
    return {
        "by_agent": rows,
        "totals": {
            "calls": sum(r["calls"] for r in rows),
            "prompt_tokens": sum(r["prompt_tokens"] for r in rows),
            "completion_tokens": sum(r["completion_tokens"] for r in rows),
            "total_tokens": sum(r["total_tokens"] for r in rows),
        },
    }


def format_report() -> str:
    """The snapshot as a console table, sorted by prompt tokens spent."""
    data = snapshot()
    rows, totals = data["by_agent"], data["totals"]
    if not rows:
        return "#### TOKEN USAGE ####\n(no LLM calls recorded)"

    sent = totals["prompt_tokens"] or 1  # only a divisor, never displayed
    header = (f"{'agent':26} {'model':22} {'calls':>5} {'sent':>9} {'%':>6} "
              f"{'avg':>7} {'max':>7} {'back':>8}")
    lines = ["#### TOKEN USAGE (sent = tokens INTO the model) ####", header,
             "-" * len(header)]
    for r in rows:
        flag = f"  ({r['calls_without_usage']} unreported)" if r["calls_without_usage"] else ""
        lines.append(
            f"{r['agent']:26} {r['model'][:22]:22} {r['calls']:>5} "
            f"{r['prompt_tokens']:>9,} {100 * r['prompt_tokens'] / sent:>5.1f}% "
            f"{r['avg_prompt_tokens']:>7,} {r['max_prompt_tokens']:>7,} "
            f"{r['completion_tokens']:>8,}{flag}"
        )
    lines.append("-" * len(header))
    lines.append(
        f"{'TOTAL':26} {'':22} {totals['calls']:>5} {totals['prompt_tokens']:>9,} "
        f"{'':>6} {'':>7} {'':>7} {totals['completion_tokens']:>8,}"
    )
    lines.append(f"grand total (sent + back): {totals['total_tokens']:,} tokens")
    return "\n".join(lines)
