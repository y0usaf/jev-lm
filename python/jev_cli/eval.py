"""Bits-per-token measurement against a unigram baseline over the same vocabulary.

This is the check that decides whether the contraption is a language model at all.
Measured at character granularity, Jev scored 4.19 bits/char against 4.00 for a
unigram model trained on 400 characters, so the context bought nothing. Run this to
get the same verdict at word granularity.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field

from .api import JevClient, choice
from .lm import INSTRUCTION
from .vocab import Vocab, renormalize

FLOOR = 1e-4  # probabilities come back quantized at about 1%; 1e-4 is below that.


@dataclass
class EvalReport:
    scored: int = 0
    skipped: int = 0
    jev_bits: float = 0.0
    unigram_bits: float = 0.0
    uniform_bits: float = 0.0
    seconds: float = 0.0
    rows: list[tuple[str, str, float, float]] = field(default_factory=list)

    @property
    def bits_per_token(self) -> float:
        return self.jev_bits / self.scored if self.scored else 0.0

    @property
    def unigram_per_token(self) -> float:
        return self.unigram_bits / self.scored if self.scored else 0.0

    @property
    def uniform_per_token(self) -> float:
        return self.uniform_bits / self.scored if self.scored else 0.0

    def table(self) -> str:
        lines = [
            f"{'context':<44} {'actual':>10} {'jev p':>8} {'unigram p':>10}",
        ]
        for context, actual, jev_p, unigram_p in self.rows:
            lines.append(f"...{context[-39:]:<41} {actual:>10} {jev_p:>8.4f} {unigram_p:>10.4f}")
        lines.append("")
        lines.append(f"scored {self.scored} positions, skipped {self.skipped} out-of-vocabulary")
        lines.append(f"  Jev      {self.jev_bits:8.2f} bits, {self.bits_per_token:.2f} bits/token")
        lines.append(f"  unigram  {self.unigram_bits:8.2f} bits, {self.unigram_per_token:.2f} bits/token")
        lines.append(f"  uniform  {self.uniform_bits:8.2f} bits, {self.uniform_per_token:.2f} bits/token")
        verdict = "context is carrying information" if self.bits_per_token < self.unigram_per_token - 0.05 else "context is NOT beating a unigram table"
        lines.append(f"  verdict: {verdict}")
        return "\n".join(lines)


def unigram_counts(tokens: list[str], vocab: Vocab) -> Counter:
    return Counter(token for token in tokens if token in vocab.tokens)


def run(
    client: JevClient,
    vocab: Vocab,
    text: str,
    corpus_text: str,
    positions: int = 40,
    rows: int = 6,
    verbose: bool = False,
) -> EvalReport:
    tokens = vocab.tokenize(text)
    corpus = unigram_counts(
        [token for token in vocab.tokenize(corpus_text) if token], vocab
    )
    corpus_total = sum(corpus.values())
    report = EvalReport()
    uniform = math.log2(len(vocab.tokens))

    for index in range(1, min(len(tokens), positions + 1)):
        actual = tokens[index]
        if actual is None:
            report.skipped += 1
            continue
        prefix = [token for token in tokens[:index] if token]
        answers = client.ask(
            {"text_so_far": vocab.render(prefix)},
            {"next": choice(INSTRUCTION, vocab.options())},
        )
        probs = answers["next"]["probabilities"]
        jev_p = max(probs.get(actual, 0.0), FLOOR)
        unigram_p = (corpus[actual] + 0.5) / (corpus_total + 0.5 * len(vocab.tokens))
        report.jev_bits += -math.log2(jev_p)
        report.unigram_bits += -math.log2(unigram_p)
        report.uniform_bits += uniform
        report.scored += 1
        if len(report.rows) < rows:
            report.rows.append((vocab.render(prefix), actual, jev_p, unigram_p))
        if verbose:
            print(f"  {report.scored:4}  {actual:>10}  jev p={jev_p:.4f}  unigram p={unigram_p:.4f}")
    return report
