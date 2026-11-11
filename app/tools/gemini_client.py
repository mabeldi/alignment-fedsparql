"""Chat completions against Google Gemini.

Gemini exposes an OpenAI-compatible API, so we reuse the OpenAI client and
just point it at Google's endpoint — exactly the same trick as
tools/ollama_client.py. That means the agents send the identical
[{"role": ..., "content": ...}] message list to Gemini and to a local model,
and switching between them is a one-line change in agent_models.yaml.

Needs GEMINI_API_KEY in app/.env.

DIFFERENCES FROM OLLAMA WORTH KNOWING
-------------------------------------
* No `reasoning_effort`. That flag is Ollama's, and llm_router only forwards it
  to providers listed in _SUPPORTS_REASONING_EFFORT (ollama only). Gemini
  manages its own thinking budget, and the 4096-token truncation problem
  described in ollama_client.py does not apply here — the context window is
  large enough that agent 7's prompt is nowhere near it.
* json_mode uses the same OpenAI `response_format` flag and is honoured.
* An empty answer is still an ERROR, not "". Returning "" made a truncated
  response look like a model with nothing to say, and the real cause was
  invisible several call frames away.

REQUESTS PER MINUTE
-------------------
The account allows 15 requests a minute and one question spends about twelve:
one to extract, one per keyword to filter matches, one per ontology to filter
paths, one to write the query. Agent 6's are issued together, so a single
question can empty the minute's budget in one burst.

Measured on "which workshops belong to a conference, and when is their
submission deadline?": agents 6 and 7 recorded ZERO calls, the run finished in
9s instead of 38s, and the harness saved an empty query with error=None. Agent 6
fails open, so the failure arrived as a full unfiltered payload rather than as an
error — the run looked like a model that had nothing to say. Re-running the same
question unchanged succeeded.

So calls are paced against a rolling 60-second window shared by every agent, and
a refused call is retried rather than lost. Pacing is the important half: a
retry after the fact still spends a slot that some other agent then goes without.
"""
import asyncio
import os
import random
import threading
import time
from collections import deque

from openai import OpenAI, AsyncOpenAI
from dotenv import load_dotenv
from tools.token_usage import record

load_dotenv()

# Google's OpenAI-compatible endpoint. Overridable via env var for testing.
_BASE_URL = os.getenv(
    "GEMINI_BASE_URL",
    "https://generativelanguage.googleapis.com/v1beta/openai/",
)
# Only a fallback — agent_models.yaml names the model for every real call.
_DEFAULT_MODEL = "gemini-3.6-flash"

_API_KEY = os.getenv("GEMINI_API_KEY")


# Requests allowed per rolling minute, and how many times a refused call is
# retried before it is allowed to fail. Both overridable for a different plan.
_RPM = int(os.getenv("GEMINI_RPM", "15"))
_MAX_RETRIES = int(os.getenv("GEMINI_MAX_RETRIES", "4"))


class _RateLimiter:
    """Hands out at most `limit` slots per rolling 60 seconds.

    A caller reserves its slot under the lock and is told how long to wait
    before using it, so the waiting happens outside the lock and concurrent
    callers queue up instead of all sleeping for the same instant and then
    firing together — which is the burst this exists to prevent.

    One instance is shared by the sync and async paths: the quota is per
    account, and counting the two separately would allow twice the limit.
    """

    def __init__(self, limit: int):
        self.limit = max(1, limit)
        self._slots: deque = deque()
        self._lock = threading.Lock()

    def reserve(self) -> float:
        """Claim the next slot; returns the seconds to wait before using it."""
        with self._lock:
            now = time.monotonic()
            while self._slots and self._slots[0] <= now - 60:
                self._slots.popleft()
            if len(self._slots) < self.limit:
                self._slots.append(now)
                return 0.0
            # The window is full: the next free moment is 60s after its oldest
            # entry. Take that moment as this call's slot and drop the entry it
            # replaces, so the window keeps exactly `limit` timestamps.
            at = self._slots[0] + 60
            self._slots.popleft()
            self._slots.append(at)
            return max(0.0, at - now)

    def wait(self):
        delay = self.reserve()
        if delay:
            time.sleep(delay)

    async def await_slot(self):
        delay = self.reserve()
        if delay:
            await asyncio.sleep(delay)


_limiter = _RateLimiter(_RPM)


def _is_rate_limited(exc: Exception) -> bool:
    """True for a refusal worth retrying, false for a real error.

    Matched on the status code rather than the exception class: the OpenAI SDK
    is being pointed at Google's endpoint, so which class surfaces for a 429 is
    not something this module should depend on.
    """
    status = getattr(exc, "status_code", None) or getattr(exc, "code", None)
    if status == 429 or str(status) == "429":
        return True
    text = str(exc).lower()
    return "429" in text or "rate limit" in text or "resource_exhausted" in text


def _backoff(attempt: int) -> float:
    """Seconds to wait after a refusal: 5, 10, 20, 40 … plus a little jitter.

    Jittered because agent 6's calls are refused at the same instant, and
    retrying them all on the same schedule reproduces the burst that caused it.
    """
    return min(60.0, 5.0 * (2 ** attempt)) + random.uniform(0, 1.5)


class MissingApiKey(RuntimeError):
    """GEMINI_API_KEY is not set — say so here rather than failing with a 401."""


class EmptyCompletion(RuntimeError):
    """The model returned no answer text. Message says why, so callers can log it."""


def _require_key():
    if not _API_KEY:
        raise MissingApiKey(
            "GEMINI_API_KEY is not set. Add it to app/.env, or switch the agent "
            "back to 'provider: ollama' in config/agent_models.yaml."
        )
    return _API_KEY


# Clients are built lazily so that merely importing this module (llm_router
# imports every provider at startup) does not fail when the key is absent and
# the user is running fully local.
_client = None
_aclient = None


def _sync_client():
    global _client
    if _client is None:
        _client = OpenAI(api_key=_require_key(), base_url=_BASE_URL)
    return _client


def _async_client():
    global _aclient
    if _aclient is None:
        _aclient = AsyncOpenAI(api_key=_require_key(), base_url=_BASE_URL)
    return _aclient


def _kwargs(json_mode):
    return {"response_format": {"type": "json_object"}} if json_mode else {}


def _content(response, model):
    """The answer text, or raise with enough detail to diagnose the failure."""
    # Before the empty-answer check: a truncated call still cost its prompt
    # tokens, and those are exactly the ones worth seeing in the report.
    record("gemini", model, getattr(response, "usage", None))

    choice = response.choices[0]
    content = (choice.message.content or "").strip()
    if content:
        return content

    reason = getattr(choice, "finish_reason", None)
    hint = ""
    if reason == "length":
        hint = (" The output limit was reached before the answer finished"
                " — shorten the prompt or raise max_tokens.")
    elif reason == "content_filter":
        hint = " Google's safety filter blocked the response."
    raise EmptyCompletion(
        f"{model} returned no content (finish_reason={reason}).{hint}"
    )


def gemini_execute(messages, temperature=0.2, model=_DEFAULT_MODEL,
                   json_mode=False):
    for attempt in range(_MAX_RETRIES + 1):
        _limiter.wait()
        try:
            response = _sync_client().chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                **_kwargs(json_mode),
            )
        except Exception as exc:
            if attempt >= _MAX_RETRIES or not _is_rate_limited(exc):
                raise
            delay = _backoff(attempt)
            print(f"[gemini] rate limited on {model} — retrying in {delay:.1f}s "
                  f"({attempt + 1}/{_MAX_RETRIES})")
            time.sleep(delay)
            continue
        return _content(response, model)


async def gemini_async_execute(messages, temperature=0.2, model=_DEFAULT_MODEL,
                               json_mode=False):
    for attempt in range(_MAX_RETRIES + 1):
        await _limiter.await_slot()
        try:
            response = await _async_client().chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                **_kwargs(json_mode),
            )
        except Exception as exc:
            if attempt >= _MAX_RETRIES or not _is_rate_limited(exc):
                raise
            delay = _backoff(attempt)
            print(f"[gemini] rate limited on {model} — retrying in {delay:.1f}s "
                  f"({attempt + 1}/{_MAX_RETRIES})")
            await asyncio.sleep(delay)
            continue
        return _content(response, model)
