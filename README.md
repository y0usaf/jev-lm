# Jev-LM

A word-level language model whose output layer is Jev. Jev never emits text, so
everything an LLM normally hides lives in this code: the tokenizer, the sampler, the
repetition mask, the stop rule, and a verified-chunk path that stands in for a KV cache.

One HTTP request per round trip, zero dependencies, Node 18 or newer.

```bash
npx jev-lm gen "she went to the door" --draft --trace
```

```
  4     nothing  p=0.024  end=0.13  done=0.92  chunk="and looked at the street" at 0.38  masked=213  0.00s
  9      looked  p=0.020  end=0.27  done=0.92  best-chunk=0.12  masked=203  stop=done  0.25s

she went to the door and looked at the street

tokens=9 round-trips=2 tokens/round-trip=4.50 stop=done
2 calls, 3700 in, 3492 out, 0.6s
```

## Install

```bash
npx jev-lm --help          # no install
npm install -g jev-lm      # global
npm install jev-lm         # library
```

Set a key one of three ways: `TYPESAFE_API_KEY`, a file named by `TYPESAFE_API_KEY_FILE`,
or `~/Tokens/TYPESAFE_API_KEY.txt`.

## Commands

```bash
jev-lm probe "the little dog was" --top 5
jev-lm verify "the capital of france is" "paris" "london"
jev-lm gen "the little dog" --words 18 --trace
jev-lm gen "she went to the door" --draft --trace
jev-lm eval --file data/heldout_en.txt --positions 60
```

`probe` shows the next-word distribution. `verify` scores candidate continuations with
Noul questions. `gen` generates. `eval` measures bits per token against a unigram
baseline over the same vocabulary.

As a library:

```ts
import { JevClient, Vocab, WordLM, Drafter } from "jev-lm";

const client = new JevClient();
const vocab = Vocab.load();
const lm = new WordLM(client, vocab, {
  maxTokens: 20,
  drafter: Drafter.fromFile("data/corpus_en.txt", vocab),
});
const generation = await lm.generate("the little dog");
console.log(generation.text, generationReport(generation));
console.log(client.usage.toString());
```

Answers are cached under `XDG_CACHE_HOME` or `~/.cache/jev-lm`, keyed by a hash of the
state, the questions, and the model, so re-running an eval costs nothing. `--no-cache`
turns that off.

## Measured results

Live calls against `api.typesafe.ai/v1/systemone`, model `jev-1.13.0`:

| Test | Result |
| --- | --- |
| Round trip, 229-option Choice plus a Noul | 0.25s median, 0.30s max |
| Round trip, plus 6 chunk Nouls | 0.27s, so question count is free in wall clock |
| Cost per round trip, 229 options | ~1,700 in tokens, ~1,700 out tokens |
| Tokens per generated word, word path | 1.10 to 1.25, so ~3,400 tokens per word |
| Tokens per generated word, `--draft` | 4.50 when the verifier accepts a chunk |
| bits/token on the shipped held-out file | Jev 6.92, unigram 6.18, uniform 7.84 |
| `done` Noul on an already complete prompt | 0.92, which is why `min_new` exists |

The bits/token row is the verdict that matters: Jev does not beat a unigram table over
the same 229 words. The baseline is favorable to the baseline, and both numbers are
inflated by the tiny vocabulary, but the ordering says the context is not carrying its
weight. At character granularity it is worse still, 4.19 bits/char against 4.00.

## Design rules, and the measurement behind each

**Mask the repetition.** Given text that ends in a period or a space, Jev puts most of
its mass on that same token. Greedy generation produced `the little dog was big . . . . .`
until the mask forbade repeating a bigram already present. Masking, then renormalizing,
is the difference between a loop and a sentence.

**Two stop signals, never one.** The `<end>` option maxed at 0.21 over 24 character
steps, so it fires on noise, and it must never reach the output text. The `done` Noul
scored 0.92 on `she went to the door`, which is correct and also means a prompt that is
already a sentence would stop generation at step zero. `min_new` requires one generated
token before either signal can stop, and `<end>` only stops when `done` agrees.

**Verify chunks instead of spelling them.** One character per round trip costs 0.25s per
character. One request that asks six Nouls about whole candidate chunks costs the same
0.25s, and the verifier separates real continuations from invented ones: `paris` 0.95
against `london` 0.02. The drafter is a local n-gram table, so it is free.

**Keep the tolerance, not the ratio.** Nested candidates from one drafter chain score
nearly flat: `and` 0.44, `and looked` 0.42, `and looked at the street` 0.38. Requiring
the top candidate to lead would reject the whole chain and emit one token. Keeping every
candidate within `keep` of the best, above an absolute floor, then taking the longest, is
what turns one round trip into five tokens.

**Treat probabilities as sparse.** A 229-option Choice assigns 0.0 to most options and
quantizes the rest at about 1 percent. Anything reading `probabilities` needs a floor,
and temperature scaling mostly reshuffles the top few.

## Layout

```
src/api.ts     HTTP client, retries, token accounting, answer cache
src/vocab.ts   vocabulary, rendering, tokenizing, masking rules
src/lm.ts      the generation loop, one request per step
src/draft.ts   n-gram drafter and chunk verification
src/eval.ts    bits/token against a unigram baseline
src/cli.ts     probe, verify, gen, eval
data/          vocabulary, drafter corpus, held-out text
python/        the original Python prototype this was ported from
```

The vocabulary ships at 229 options. A Choice accepts 255 and TypeSafe documents
reliable behavior up to roughly 240, so `--vocab` will take a slightly larger file than
the default, not a much larger one.

## What this is not

Not a text generator to ship in a product. The vocabulary is 229 words, the measured
quality loses to a unigram table at both character and word granularity, and the word
path costs roughly 3,400 tokens per word. The reason to keep it around is the inverse of
the reason to write a wrapper: the interesting parts of an LLM turn out to be the parts
you had to write by hand here.

MIT.
