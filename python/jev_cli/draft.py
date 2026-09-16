"""N-gram drafter. Proposes whole chunks that a Noul can accept or reject.

The measured reason this exists: one round trip costs the same whether you ask one
question or six, because questions inside a request run in parallel. So a free local
drafter plus one verification request per accepted chunk buys several tokens per round
trip instead of one.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from .vocab import END, PUNCTUATION, Vocab


@dataclass
class Drafter:
    vocab: Vocab
    order: int = 3
    max_len: int = 8
    counts: dict[tuple[str, ...], Counter] = field(default_factory=lambda: defaultdict(Counter))

    @classmethod
    def from_text(cls, text: str, vocab: Vocab, order: int = 3, max_len: int = 8) -> "Drafter":
        drafter = cls(vocab=vocab, order=order, max_len=max_len)
        drafter.add(text)
        return drafter

    @classmethod
    def from_file(cls, path: str | Path, vocab: Vocab, order: int = 3, max_len: int = 8) -> "Drafter":
        """One document per line, so proposed chunks never cross a sentence break."""
        drafter = cls(vocab=vocab, order=order, max_len=max_len)
        for line in Path(path).read_text().splitlines():
            if line.strip() and not line.lstrip().startswith("#"):
                drafter.add(line)
        return drafter

    def add(self, text: str) -> None:
        tokens = [token for token in self.vocab.tokenize(text) if token]
        for width in range(1, self.order + 1):
            key_width = width - 1
            for index in range(len(tokens) - width + 1):
                key = tuple(tokens[index : index + key_width])
                self.counts[key][tokens[index + key_width]] += 1

    def next_token(self, context: tuple[str, ...], min_width: int = 0) -> str | None:
        """Longest-match backoff: try the full context, then drop the oldest token.

        min_width stops the backoff before it reaches the empty context. Chunk
        extension uses min_width=1 so a chain cannot jump to an unrelated sentence.
        """
        for width in range(min(len(context), self.order - 1), min_width - 1, -1):
            key = context[len(context) - width :] if width else ()
            counter = self.counts.get(key)
            if counter:
                return counter.most_common(1)[0][0]
        return None

    def draw(self, context: tuple[str, ...], limit: int = 2) -> list[str]:
        for width in range(min(len(context), self.order - 1), -1, -1):
            key = context[len(context) - width :] if width else ()
            counter = self.counts.get(key)
            if counter:
                return [token for token, _ in counter.most_common(limit)]
        return []

    def top_tokens(self, tokens: list[str], k: int = 40) -> list[str]:
        """The k most likely next tokens, by backoff from the longest context to the
        unigram table. This is the candidate set for a narrowed Choice question.

        Measured on held-out text: the true next token is inside the top 4 for 0.50 of
        positions, top 12 for 0.67, top 20 for 0.73, top 40 for 0.81, top 80 for 0.90.
        A narrow option map is a bet that pays off about four times in five at k=40.
        """
        out: list[str] = []
        for width in range(min(len(tokens), self.order - 1), -1, -1):
            key = tuple(tokens[len(tokens) - width :]) if width else ()
            counter = self.counts.get(key)
            if counter:
                for token, _ in counter.most_common(k):
                    if token not in out:
                        out.append(token)
            if len(out) >= k:
                break
        return out[:k]

    def candidates(self, tokens: list[str], k: int = 6) -> list[str]:
        """Whole continuations to verify, longest first, up to k of them."""
        context = tuple(tokens)
        chains: list[list[str]] = []
        for first in self.draw(context, limit=2):
            chain = [first]
            chains.append(list(chain))
            for _ in range(self.max_len - 1):
                following = self.next_token(context + tuple(chain), min_width=1)
                if following is None or following == END:
                    break
                chain.append(following)
                chains.append(list(chain))

        rendered: list[str] = []
        for chain in sorted(chains, key=len, reverse=True):
            text = " ".join(chain)
            if text not in rendered:
                rendered.append(text)
        return rendered[:k]


def verified_chunk(
    answers: dict,
    candidates: list[str],
    accept: float = 0.25,
    keep: float = 0.8,
) -> tuple[str, list[tuple[str, float]]]:
    """Read the chunk Nouls out of one request and return the chunk to insert, plus scores.

    Rule: keep the candidates scoring at least keep times the best score, above an
    absolute accept floor, then take the longest survivor. Measured behavior:

    * Unrelated candidates separate well: 'Paris' 0.95 against 'London' 0.02.
    * Nested candidates barely separate, and the shortest wins slightly: 'and' 0.44,
      'and looked' 0.42, 'and looked at the street' 0.38. A length-agnostic rule picks
      one token; keep=0.8 picks the whole chain, which is the point of the drafter.
    * Four invented continuations all landed at 0.14 to 0.17, below any sane floor.
    """
    scored = [
        (candidate, float(answers[f"chunk_{index}"]["noul"]))
        for index, candidate in enumerate(candidates)
        if answers.get(f"chunk_{index}", {}).get("type") == "noul"
    ]
    if not scored:
        return "", []
    top = max(value for _, value in scored)
    threshold = max(accept, top * keep)
    qualified = [(candidate, value) for candidate, value in scored if value >= threshold]
    if not qualified:
        return "", scored
    best = max(qualified, key=lambda pair: (count_words(pair[0]), pair[1]))[0]
    return best, sorted(scored, key=lambda pair: (-pair[1], -len(pair[0])))


def count_words(candidate: str) -> int:
    words = [word for word in candidate.split() if word not in PUNCTUATION]
    return len(words or candidate.split())
