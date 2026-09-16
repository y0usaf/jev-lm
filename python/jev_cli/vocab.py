"""Vocabulary loading, rendering, tokenizing, and the option-masking rules."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"
DEFAULT_VOCAB = DATA_DIR / "vocab_en.txt"
DEFAULT_CORPUS = DATA_DIR / "corpus_en.txt"
END = "<end>"
NONE = "<none>"
PUNCTUATION = {".", ",", "?", "!", ":", ";", "...", "-"}
END_DESCRIPTION = "nothing should follow; the text is finished"
NONE_DESCRIPTION = "the next token is not among the options above"


@dataclass(frozen=True)
class Vocab:
    tokens: tuple[str, ...]

    @classmethod
    def load(cls, path: str | Path | None = None) -> "Vocab":
        source = Path(path) if path else DEFAULT_VOCAB
        tokens = [
            line.strip()
            for line in source.read_text().splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]
        if len(tokens) > 255:
            raise ValueError(f"{source} has {len(tokens)} options; a Choice accepts at most 255")
        return cls(tuple(tokens))

    def options(self, allowed: set[str] | None = None) -> dict[str, str | None]:
        """The criteria map for a Choice question. Keep descriptions null where the
        name speaks for itself: every option costs tokens on every call.

        Pass `allowed` to offer only those tokens. The option count is the cost of the
        call in both directions: 229 options measured 1706 in and 1638 out tokens, an
        11-option map measured 364 in and 98 out on the same state.
        """
        tokens = self.tokens if allowed is None else tuple(t for t in self.tokens if t in allowed)
        return {token: (END_DESCRIPTION if token == END else None) for token in tokens}

    def render(self, tokens: list[str] | tuple[str, ...]) -> str:
        """Join tokens into readable text. Punctuation sticks to the previous token."""
        out = ""
        for token in tokens:
            if not out:
                out = token
            elif token in PUNCTUATION:
                out += token
            else:
                out += " " + token
        return out

    def tokenize(self, text: str) -> list[str | None]:
        """Longest-match tokenize against this vocabulary. None marks a word with no option."""
        words = text.split()
        tokens: list[str | None] = []
        index = 0
        widest = max(len(token.split()) for token in self.tokens)
        while index < len(words):
            for width in range(widest, 0, -1):
                candidate = " ".join(words[index : index + width])
                canonical = candidate.lower()
                if canonical in self.tokens:
                    tokens.append(canonical)
                    index += width
                    break
                if candidate in self.tokens:
                    tokens.append(candidate)
                    index += width
                    break
            else:
                tokens.append(None)
                index += 1
        return tokens

    def allowed(
        self,
        tokens: list[str],
        min_tokens: int = 0,
        ban_repeat_bigrams: bool = True,
    ) -> set[str]:
        """Options that survive the anti-degeneracy mask.

        Measured failure mode: given a text ending in a space or a period, Jev puts
        most of its mass on repeating that character or word. Masking the repeat in
        code, then renormalizing, is what breaks the loop.
        """
        allowed = set(self.tokens)
        if len(tokens) < min_tokens:
            allowed.discard(END)
        if tokens:
            allowed.discard(tokens[-1])
        if ban_repeat_bigrams and len(tokens) >= 2:
            seen = set(zip(tokens, tokens[1:]))
            allowed = {token for token in allowed if (tokens[-1], token) not in seen}
        return allowed


def renormalize(probs: dict[str, float], allowed: set[str] | None) -> dict[str, float]:
    """Drop masked options and rescale. Falls back to the raw distribution if empty."""
    if allowed is None:
        return dict(probs)
    kept = {token: p for token, p in probs.items() if token in allowed and p > 0.0}
    total = sum(kept.values())
    if total <= 0.0:
        return dict(probs)
    return {token: p / total for token, p in kept.items()}
