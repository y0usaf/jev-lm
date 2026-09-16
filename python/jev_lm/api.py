"""HTTP client for the TypeSafe System One endpoint. Standard library only."""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ENDPOINT = "https://api.typesafe.ai/v1/systemone"
DEFAULT_MODEL = "jev-latest"
DEFAULT_KEY_PATH = "~/Tokens/TYPESAFE_API_KEY.txt"
RETRY_STATUS = {408, 409, 429, 500, 502, 503, 504, 529}

Question = dict[str, Any]


class JevError(RuntimeError):
    """The API refused a call, stayed unreachable, or the key is missing."""


def api_key(explicit: str | None = None, path: str | None = None) -> str:
    """Resolve the key from an argument, the environment, then a file."""
    if explicit:
        return explicit.strip()
    from_env = os.environ.get("TYPESAFE_API_KEY")
    if from_env:
        return from_env.strip()
    candidate = Path(path or os.environ.get("TYPESAFE_API_KEY_FILE", DEFAULT_KEY_PATH)).expanduser()
    if candidate.is_file():
        key = candidate.read_text().strip()
        if key:
            return key
    raise JevError(f"no API key found: pass one, set TYPESAFE_API_KEY, or write one to {candidate}")


@dataclass
class Usage:
    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    seconds: float = 0.0
    cached: int = 0

    def add(self, input_tokens: int, output_tokens: int, seconds: float) -> None:
        self.calls += 1
        self.input_tokens += input_tokens
        self.output_tokens += output_tokens
        self.seconds += seconds

    def __str__(self) -> str:
        return (
            f"{self.calls} calls, {self.input_tokens} in, {self.output_tokens} out, "
            f"{self.seconds:.1f}s" + (f", {self.cached} cached" if self.cached else "")
        )


def _digest(state: Any, questions: dict[str, Question], model: str) -> str:
    payload = json.dumps({"state": state, "questions": questions, "model": model}, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()[:24]


class JevClient:
    """One method, one round trip per call. Timing and token use accumulate on .usage."""

    def __init__(
        self,
        key: str | None = None,
        key_path: str | None = None,
        model: str = DEFAULT_MODEL,
        timeout: float = 120.0,
        retries: int = 4,
        max_calls: int = 0,
        cache_dir: str | None = "~/.cache/jev-lm",
        verbose: bool = False,
    ) -> None:
        self._key = api_key(key, key_path)
        self.model = model
        self.timeout = timeout
        self.retries = retries
        self.max_calls = max_calls
        self.verbose = verbose
        self.cache_dir = Path(cache_dir).expanduser() if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.usage = Usage()

    def ask(self, state: Any, questions: dict[str, Question]) -> dict[str, Any]:
        """Send one request and return the answers map."""
        if self.max_calls and self.usage.calls >= self.max_calls:
            raise JevError(f"call budget of {self.max_calls} reached")

        cache_path = None
        if self.cache_dir:
            cache_path = self.cache_dir / f"{_digest(state, questions, self.model)}.json"
            if cache_path.is_file():
                self.usage.cached += 1
                return json.loads(cache_path.read_text())["answers"]

        body = json.dumps({"state": state, "model": self.model, "questions": questions}).encode()
        last_error: Exception | None = None
        for attempt in range(self.retries + 1):
            request = urllib.request.Request(
                ENDPOINT,
                data=body,
                headers={"Authorization": f"Bearer {self._key}", "Content-Type": "application/json"},
            )
            started = time.time()
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    payload = json.loads(response.read())
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode(errors="replace")[:400]
                last_error = JevError(f"HTTP {exc.code} from {ENDPOINT}: {detail}")
                if exc.code not in RETRY_STATUS:
                    raise last_error from exc
            except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
                last_error = JevError(f"{type(exc).__name__} on {ENDPOINT}: {exc}")
            else:
                seconds = time.time() - started
                usage = payload.get("usage", {})
                self.usage.add(
                    int(usage.get("input_tokens", 0)), int(usage.get("output_tokens", 0)), seconds
                )
                if self.verbose:
                    print(
                        f"  [jev] {len(questions)} questions, {seconds:.2f}s, "
                        f"{usage.get('input_tokens', 0)} in, {usage.get('output_tokens', 0)} out"
                    )
                if cache_path:
                    cache_path.write_text(json.dumps({"answers": payload["answers"]}))
                return payload["answers"]
            time.sleep(min(2.0 * (2**attempt), 20.0))
        raise last_error or JevError("request failed")


def noul(instructions: str, criteria: dict[str, str] | None = None) -> Question:
    question: Question = {"type": "noul", "instructions": instructions}
    if criteria:
        question["criteria"] = criteria
    return question


def choice(instructions: str, options: dict[str, str | None]) -> Question:
    return {"type": "choice", "instructions": instructions, "criteria": options}


def quote(text: str) -> str:
    """Single-quote a chunk for embedding in a Noul instruction."""
    return "'" + re.sub(r"\s+", " ", text).strip().replace("'", "\\'") + "'"
