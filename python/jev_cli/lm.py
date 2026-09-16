"""Word-level generation: one round trip per step, with masking, sampling, and verification."""

from __future__ import annotations

import random
import time
from dataclasses import dataclass, field

from .api import JevClient, choice, noul, quote
from .draft import Drafter, count_words, verified_chunk
from .vocab import END, NONE, NONE_DESCRIPTION, PUNCTUATION, Vocab, renormalize

INSTRUCTION = "Which word comes next in `text_so_far`?"
DONE_INSTRUCTION = "`text_so_far` is already a complete and natural sentence as it stands."


@dataclass
class Step:
    index: int
    seconds: float
    picked: str
    picked_p: float
    end_p: float
    done: float
    top: list[tuple[str, float]]
    chunk: str = ""
    chunk_score: float = 0.0
    chunk_scores: list[tuple[str, float]] = field(default_factory=list)
    masked: int = 0
    options: int | None = None
    escalated: bool = False
    added: int = 1
    end_ignored: bool = False
    stopped: str = ""

    def line(self) -> str:
        head = (
            f"{self.index:3}  {self.picked:>10}  p={self.picked_p:.3f}  "
            f"end={self.end_p:.2f}  done={self.done:.2f}"
        )
        if self.chunk:
            head += f"  chunk={self.chunk!r} at {self.chunk_score:.2f}"
        elif self.chunk_scores:
            head += f"  best-chunk={self.chunk_scores[0][1]:.2f}"
        if self.masked:
            head += f"  masked={self.masked}"
        if self.options:
            head += f"  options={self.options}"
        if self.escalated:
            head += "  escalated"
        if self.end_ignored:
            head += "  end-ignored"
        if self.stopped:
            head += f"  stop={self.stopped}"
        return head + f"  {self.seconds:.2f}s"


@dataclass
class Generation:
    prompt: str
    tokens: list[str]
    text: str
    steps: list[Step]
    stop_reason: str

    def escalations(self) -> int:
        return sum(1 for step in self.steps if step.escalated)

    def round_trips(self) -> int:
        return len(self.steps) + self.escalations()

    def tokens_per_round_trip(self) -> float:
        return len(self.tokens) / max(self.round_trips(), 1)

    def report(self) -> str:
        return (
            f"tokens={len(self.tokens)} round-trips={self.round_trips()} "
            f"tokens/round-trip={self.tokens_per_round_trip():.2f} "
            f"escalated={self.escalations()}/{len(self.steps)} stop={self.stop_reason}"
        )


class WordLM:
    """A language model whose output layer is a Jev Choice question.

    Everything an LLM normally hides lives here: the tokenizer, the sampler, the
    repetition mask, the stop rule, and a verified-chunk path instead of a cache.
    """

    def __init__(
        self,
        client: JevClient,
        vocab: Vocab,
        temp: float = 0.7,
        top_p: float = 0.95,
        accept: float = 0.25,
        keep: float = 0.8,
        stop_done: float = 0.5,
        stop_end: float = 0.2,
        stop_agree: float = 0.25,
        min_new: int = 1,
        max_tokens: int = 40,
        min_tokens: int = 6,
        seed: int = 0,
        drafter: Drafter | None = None,
        candidates: int = 6,
        narrow: int = 0,
        verbose: bool = False,
    ) -> None:
        self.client = client
        self.vocab = vocab
        self.temp = temp
        self.top_p = top_p
        self.accept = accept
        self.keep = keep
        self.stop_done = stop_done
        self.stop_end = stop_end
        self.stop_agree = stop_agree
        self.min_new = min_new
        self.max_tokens = max_tokens
        self.min_tokens = min_tokens
        self.drafter = drafter
        self.candidates = candidates
        self.narrow = narrow
        self.verbose = verbose
        self.random = random.Random(seed)

    def sample(self, probs: dict[str, float]) -> tuple[str, float]:
        if self.temp <= 0.0:
            pick = max(probs, key=lambda token: probs[token])
            return pick, probs[pick]
        scaled = sorted(
            ((token, value ** (1.0 / self.temp)) for token, value in probs.items()),
            key=lambda pair: -pair[1],
        )
        total = sum(value for _, value in scaled)
        cumulative, kept = 0.0, []
        for token, value in scaled:
            weight = value / total
            kept.append((token, weight))
            cumulative += weight
            if cumulative >= self.top_p:
                break
        mass = sum(weight for _, weight in kept)
        pick = self.random.choices([token for token, _ in kept], weights=[w / mass for _, w in kept])[0]
        return pick, probs[pick]

    def criteria(self, tokens: list[str], allowed: set[str]) -> tuple[dict[str, str | None], bool]:
        """The option map for this step's Choice question, and whether it was narrowed.

        A Choice answers over the whole criteria map and every option costs tokens both
        ways, so the map is the cost of a step: 229 options is ~3.3k tokens, tens of
        options is ~1k. `narrow` offers the drafter's top candidates plus punctuation,
        `<end>`, and a `<none>` escape hatch, and the step buys the full map back only
        when Jev takes `<none>`.
        """
        if not (self.narrow and self.drafter):
            return self.vocab.options(), False
        offered = [token for token in self.drafter.top_tokens(tokens, k=self.narrow) if token in allowed]
        offered.extend(token for token in sorted(PUNCTUATION) if token in allowed)
        if END in allowed:
            offered.append(END)
        criteria = self.vocab.options(set(offered))
        criteria[NONE] = NONE_DESCRIPTION
        return criteria, True

    def step(self, tokens: list[str]) -> Step:
        text = self.vocab.render(tokens)
        allowed = self.vocab.allowed(tokens, min_tokens=self.min_tokens)
        proposed = self.drafter.candidates(tokens, k=self.candidates) if self.drafter else []
        criteria, narrowed = self.criteria(tokens, allowed)

        questions = {
            "next": choice(INSTRUCTION, criteria),
            "done": noul(DONE_INSTRUCTION),
        }
        for index, candidate in enumerate(proposed):
            questions[f"chunk_{index}"] = noul(
                f"Immediately after `text_so_far` comes the text {quote(candidate)}."
            )

        started = time.time()
        answers = self.client.ask({"text_so_far": text}, questions)
        raw = answers["next"]["probabilities"]

        # The candidate set missed, so buy the full vocabulary rather than emit a token
        # the drafter never proposed. The done and chunk answers from the first call were
        # asked about the same state, so they carry over to the escalated answer.
        escalated = narrowed and max(raw, key=lambda token: raw[token]) == NONE
        if escalated:
            full = self.client.ask(
                {"text_so_far": text}, {"next": choice(INSTRUCTION, self.vocab.options())}
            )
            answers = {**answers, **full}
            raw = answers["next"]["probabilities"]
        seconds = time.time() - started
        kept = renormalize(raw, allowed)
        masked = len(raw) - len(kept)
        pick, pick_p = self.sample(kept)
        top = sorted(kept.items(), key=lambda pair: -pair[1])[:5]
        done = float(answers["done"]["noul"])
        end_p = float(kept.get(END, raw.get(END, 0.0)))
        chunk, chunk_scores = verified_chunk(answers, proposed, self.accept, self.keep)
        chunk = chunk if count_words(chunk) >= 2 else ""

        # The <end> option maxed at 0.21 over 24 character steps, so it fires on noise,
        # and it must never reach the output text. Substitute the best real token and let
        # the caller's stop rules use end_p and done together.
        end_ignored = False
        if pick == END:
            fallback = next((token for token, _ in top if token != END), None)
            if fallback is not None:
                pick, pick_p, end_ignored = fallback, kept.get(fallback, 0.0), True

        return Step(
            index=len(tokens),
            seconds=seconds,
            picked=pick,
            picked_p=pick_p,
            end_p=end_p,
            done=done,
            top=top,
            chunk=chunk,
            chunk_score=next((value for candidate, value in chunk_scores if candidate == chunk), 0.0),
            chunk_scores=chunk_scores,
            masked=masked,
            options=None if escalated or not narrowed else len(criteria),
            escalated=escalated,
            end_ignored=end_ignored,
        )

    def generate(self, prompt: str) -> Generation:
        tokens = [token for token in self.vocab.tokenize(prompt) if token]
        prompt_length = len(tokens)
        steps: list[Step] = []
        stop_reason = "max_tokens"
        while len(tokens) < self.max_tokens:
            step = self.step(tokens)
            steps.append(step)

            # The done and <end> questions were asked about the text before this step's
            # token, so the stop check has to happen before the token is appended.
            added = len(tokens) - prompt_length
            if step.done >= self.stop_done and added >= self.min_new:
                stop_reason, step.stopped = "done", "done"
                break
            if step.end_p >= self.stop_end and step.done >= self.stop_agree and added >= self.min_new:
                stop_reason, step.stopped = "end-option", "end-option"
                break

            if step.chunk:
                tokens.extend(step.chunk.split())
            else:
                tokens.append(step.picked)
            step.added = len(tokens) - step.index
        return Generation(prompt, tokens, self.vocab.render(tokens), steps, stop_reason)
