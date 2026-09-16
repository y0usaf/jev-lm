"""Command line: probe, verify, gen, eval."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .api import JevClient, JevError, choice, noul, quote
from .draft import Drafter
from .eval import run as run_eval
from .lm import INSTRUCTION, WordLM
from .vocab import DEFAULT_CORPUS, Vocab


def build(args: argparse.Namespace) -> JevClient:
    return JevClient(
        model=args.model,
        max_calls=args.max_calls,
        verbose=args.verbose,
        cache_dir=None if args.no_cache else "~/.cache/jev-lm",
    )


def cmd_probe(args: argparse.Namespace) -> int:
    vocab = Vocab.load(args.vocab)
    client = build(args)
    answers = client.ask(
        {"text_so_far": args.prompt}, {"next": choice(INSTRUCTION, vocab.options())}
    )
    answer = answers["next"]
    print(f"prompt: {args.prompt!r}")
    print(f"choice={answer['choice']!r} confidence={answer['confidence']:.3f}")
    ranked = sorted(answer["probabilities"].items(), key=lambda pair: -pair[1])
    for token, probability in ranked[: args.top]:
        print(f"  {token:>10}  {probability:.4f}")
    print(f"  p(<end>)={answer['probabilities'].get('<end>', 0.0):.4f}")
    print(client.usage)
    return 0


def cmd_verify(args: argparse.Namespace) -> int:
    client = build(args)
    questions = {
        f"chunk_{index}": noul(f"Immediately after `text_so_far` comes the text {quote(candidate)}.")
        for index, candidate in enumerate(args.candidates)
    }
    questions["done"] = noul("`text_so_far` is already a complete and natural sentence as it stands.")
    answers = client.ask({"text_so_far": args.prompt}, questions)
    for index, candidate in enumerate(args.candidates):
        print(f"  {answers[f'chunk_{index}']['noul']:.3f}  {candidate!r}")
    print(f"  {answers['done']['noul']:.3f}  <already complete>")
    print(client.usage)
    return 0


def cmd_gen(args: argparse.Namespace) -> int:
    vocab = Vocab.load(args.vocab)
    client = build(args)
    drafter = None
    if args.draft or args.narrow:
        corpus = Path(args.corpus) if args.corpus else DEFAULT_CORPUS
        drafter = Drafter.from_file(corpus, vocab, order=args.order, max_len=args.chunk_len)
    lm = WordLM(
        client,
        vocab,
        temp=args.temp,
        top_p=args.top_p,
        accept=args.accept,
        keep=args.keep,
        max_tokens=args.words,
        min_tokens=args.min_words,
        seed=args.seed,
        drafter=drafter,
        # Chunk verification only runs under --draft; --narrow alone must leave the word
        # path intact, otherwise the drafter's chunks decide the run and the option map
        # never gets measured.
        candidates=args.candidates if args.draft else 0,
        narrow=args.narrow,
        verbose=args.verbose,
    )
    generation = lm.generate(args.prompt)
    if args.trace:
        for step in generation.steps:
            print(step.line())
        print()
    print(generation.text)
    print()
    print(generation.report())
    print(client.usage)
    return 0


def cmd_eval(args: argparse.Namespace) -> int:
    vocab = Vocab.load(args.vocab)
    client = build(args)
    corpus = Path(args.corpus).read_text() if args.corpus else DEFAULT_CORPUS.read_text()
    text = Path(args.file).read_text() if args.file else corpus
    report = run_eval(
        client, vocab, text, corpus, positions=args.positions, rows=args.rows, verbose=args.verbose
    )
    print(report.table())
    print(client.usage)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="jev-lm", description="A language model whose output layer is Jev.")
    parser.add_argument("--model", default="jev-latest")
    parser.add_argument("--vocab", default=None, help="vocabulary file, one token per line")
    parser.add_argument("--corpus", default=None, help="corpus for the drafter and the unigram baseline")
    parser.add_argument("--max-calls", type=int, default=0, help="stop after this many API calls")
    parser.add_argument("--no-cache", action="store_true", help="do not reuse cached answers")
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)

    probe = sub.add_parser("probe", help="show the next-word distribution for a prompt")
    probe.add_argument("prompt")
    probe.add_argument("--top", type=int, default=12)
    probe.set_defaults(func=cmd_probe)

    verify = sub.add_parser("verify", help="score candidate continuations with Noul questions")
    verify.add_argument("prompt")
    verify.add_argument("candidates", nargs="+")
    verify.set_defaults(func=cmd_verify)

    gen = sub.add_parser("gen", help="generate text one word per round trip")
    gen.add_argument("prompt")
    gen.add_argument("--words", type=int, default=30)
    gen.add_argument("--min-words", type=int, default=6)
    gen.add_argument("--temp", type=float, default=0.7)
    gen.add_argument("--top-p", type=float, default=0.95)
    gen.add_argument("--seed", type=int, default=0)
    gen.add_argument("--trace", action="store_true")
    gen.add_argument("--draft", action="store_true", help="propose chunks with an n-gram drafter and verify them")
    gen.add_argument("--accept", type=float, default=0.25, help="absolute Noul floor for a chunk")
    gen.add_argument("--keep", type=float, default=0.8, help="fraction of the best chunk score a candidate must retain")
    gen.add_argument("--candidates", type=int, default=6)
    gen.add_argument(
        "--narrow",
        type=int,
        default=0,
        help="offer the drafter's top K candidates plus <none> instead of the full vocabulary",
    )
    gen.add_argument("--order", type=int, default=3, help="drafter n-gram order")
    gen.add_argument("--chunk-len", type=int, default=8, help="longest drafter chunk in tokens")
    gen.set_defaults(func=cmd_gen)

    evaluate = sub.add_parser("eval", help="bits per token against a unigram baseline")
    evaluate.add_argument("--file", default=None, help="held-out text; defaults to the corpus")
    evaluate.add_argument("--positions", type=int, default=40)
    evaluate.add_argument("--rows", type=int, default=6)
    evaluate.set_defaults(func=cmd_eval)

    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except JevError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
