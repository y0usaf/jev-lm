# jev-cli

A word-level language model whose output layer is a Jev `Choice` question. Jev never
emits text, so everything an LLM normally hides lives in this code: the tokenizer, the
sampler, the repetition mask, the stop rule, and a verified-chunk path that stands in
for a KV cache.

Standard library only. No SDK needed, one POST per round trip.

## Measured results

From live calls against `api.typesafe.ai/v1/systemone`, model `jev-1.13.0`, on this
machine:

| Test | Result |
| --- | --- |
| Round trip, 229-option Choice + Noul | 0.25s median, 0.30s max |
| Round trip, +6 chunk Nouls | 0.27s, so question count is free in wall clock |
| Cost per round trip, 229 options | ~1,700 in, ~1,700 out tokens |
| Cost per round trip, `--narrow 40` | ~630 in, ~360 out tokens, ~45 options offered |
| Escalation rate, `--narrow 40` | 3 of 19 steps took `<none>` and bought the full map |
| Tokens per generated word, word path | 1.10 to 1.25, so ~3,400 tokens per word |
| Tokens per generated word, `--draft` | 4.50 when the verifier accepts a chunk |
| bits/token on held-out text | Jev 6.82, unigram 6.18, uniform 7.84 |
| `done` Noul on a complete prompt | 0.92, which is why generation stops early without `min_new` |

The bits/token number is the verdict that matters: at word granularity Jev still does
not beat a unigram table trained on 336 words. The baseline is favorable to the
baseline, and both numbers are inflated by the tiny vocabulary, but the ordering says
context is not carrying its weight.

## Commands

```bash
cd ~/sandbox/jev-cli
python3 -m jev_cli probe "the capital of france is"          # next-word distribution
python3 -m jev_cli verify "the capital of france is" "paris" "london"
python3 -m jev_cli gen "the little dog" --words 18 --trace
python3 -m jev_cli gen "she went to the door" --draft --trace # n-gram drafter + Noul verification
python3 -m jev_cli eval --file jev_cli/data/heldout_en.txt --positions 60
```

The key comes from `TYPESAFE_API_KEY`, or from `~/Tokens/TYPESAFE_API_KEY.txt` by
default. Set `TYPESAFE_API_KEY_FILE` to point elsewhere. Answers are cached in
`~/.cache/jev-cli`, so re-running an eval costs nothing.

Useful flags:

- `--vocab FILE` swaps the vocabulary. Keep it under 235 lines: a Choice accepts 255
  options and TypeSafe documents reliable behavior up to roughly 240.
- `--corpus FILE` feeds the drafter and the unigram baseline. Every word in it must be
  in the vocabulary, otherwise the drafter's context keys have holes.
- `--accept 0.25 --keep 0.8` control chunk acceptance. `--temp`, `--top-p`, `--seed`
  control sampling. `--max-calls N` caps API use.
- `--narrow 40` offers the drafter's top 40 candidates plus punctuation, `<end>`, and a
  `<none>` escape hatch instead of the full vocabulary, and escalates to the full map on
  `<none>`. Needs a corpus, the same one the drafter uses. `--draft` alone still leaves
  the word path untouched.

## Design rules, and the measurement behind each

**Mask the repetition.** Given text that ends in a period or a space, Jev puts most of
its mass on that same token. Grammar-greedy generation produced `the little dog was
big . . . . .` until the mask forbade repeating a bigram already present. Masking, then
renormalizing, is the difference between a loop and a sentence.

**Two stop signals, never one.** The `<end>` option maxed at 0.21 over 24 character
steps, so it fires on noise. The `done` Noul scored 0.92 on `she went to the door`,
which is correct and also means a prompt that is already a sentence stops generation at
step zero. `min_new=4` requires four generated tokens before either signal can stop,
and `<end>` only stops when `done` agrees.

**Verify chunks instead of spelling them.** Asking for one character at a time cost
0.25s per character and produced a 4.19 bits/character model, worse than a unigram
table. One request that asks six Nouls about whole candidate chunks costs the same
0.25s, and the verifier separates real continuations from invented ones: `Paris` 0.95
against `London` 0.02. The drafter is a local n-gram table, so it is free.

**Keep the tolerance, not the ratio.** Nested candidates from one drafter chain score
nearly flat: `and` 0.44, `and looked` 0.42, `and looked at the street` 0.38. Requiring
the top candidate to lead rejects the whole chain and emits one token. Keeping every
candidate within `keep` of the best, above an absolute floor, then taking the longest,
is what turns one round trip into five tokens.

**Treat probabilities as sparse.** A 229-option Choice assigns 0.0 to most options and
quantizes the rest at about 1 percent. Everything reading `probabilities` needs a floor,
and temperature scaling on a sparse distribution mostly reshuffles the top few options.

**Narrow the option map, do not mask it.** The mask runs after the answer arrives, so
options it would drop are paid for anyway, and it only ever removes about two of 229:
`allowed` measures 227 on every context tried. The option count is the cost, roughly 6
input and 7 output tokens per option. So offer fewer instead: the drafter's top 40 plus
punctuation, `<end>`, and a `<none>` escape hatch is about 45 options and 1.0k tokens per
step against 3.4k for the full map, and a step that answers `<none>` buys the full map as
a second request. Top 40 covers the true next token 0.81 of the time on this held-out
text, 3 of 19 steps escalated, and the word path came out 2.3x to 2.6x cheaper end to
end. `--narrow K` opts in. The default is still the full map because `eval` does not
score the narrowed question yet, so its effect on bits/token is unmeasured.

## Files

```
jev_cli/api.py     HTTP client, retries, token accounting, answer cache
jev_cli/vocab.py   vocabulary, rendering, tokenizing, masking rules
jev_cli/lm.py      the generation loop: one request per step
jev_cli/draft.py   n-gram drafter, chunk verification, narrowed-option candidates
jev_cli/eval.py    bits/token against a unigram baseline
jev_cli/data/      vocabulary, drafter corpus, held-out text
```

## What this is not

It is not a text generator you would ship. The vocabulary is 229 words, the measured
quality loses to a unigram table at both character and word granularity, and the cost
is roughly 3,400 tokens per word on the word path. The reason to keep it is the
inverse of the reason to write a wrapper: the interesting parts of an LLM turn out to be
the parts you had to write by hand here.
