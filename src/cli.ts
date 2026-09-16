#!/usr/bin/env node
/**
 * Command line: probe, verify, gen, eval.
 *
 * Flags may appear before or after the command. No argument-parsing dependency is used.
 */

import { readFileSync } from "node:fs";

import { JevClient, JevError, choice, noul, quote } from "./api.js";
import type { Questions } from "./api.js";
import { Drafter } from "./draft.js";
import { evalTable, runEval } from "./eval.js";
import { DONE_INSTRUCTION, INSTRUCTION, WordLM, generationReport, stepLine } from "./lm.js";
import { DEFAULT_CORPUS, Vocab } from "./vocab.js";

type FlagSpec = Record<string, "string" | "boolean">;

interface Parsed {
  command: string;
  positionals: string[];
  flags: Map<string, string | boolean>;
}

const GLOBAL_FLAGS: FlagSpec = {
  "--model": "string",
  "--vocab": "string",
  "--corpus": "string",
  "--max-calls": "string",
  "--no-cache": "boolean",
  "--verbose": "boolean",
  "-v": "boolean",
  "--help": "boolean",
  "-h": "boolean",
  "--version": "boolean",
};

const COMMAND_FLAGS: Record<string, FlagSpec> = {
  probe: { "--top": "string" },
  verify: {},
  gen: {
    "--words": "string",
    "--min-words": "string",
    "--temp": "string",
    "--top-p": "string",
    "--seed": "string",
    "--trace": "boolean",
    "--draft": "boolean",
    "--accept": "string",
    "--keep": "string",
    "--candidates": "string",
    "--order": "string",
    "--chunk-len": "string",
  },
  eval: { "--file": "string", "--positions": "string", "--rows": "string" },
};

const USAGE = `Jev-LM: a word-level language model whose output layer is Jev

Usage
  jev-lm [global flags] <command> [flags] [args]

Commands
  probe <prompt>               show the next-word distribution for a prompt
  verify <prompt> <chunk...>   score candidate continuations with Noul questions
  gen <prompt>                 generate text, one word or one verified chunk per round trip
  eval                         bits per token against a unigram baseline

Global flags
  --model NAME     model id, default jev-latest
  --vocab FILE     vocabulary file, one token per line, under 235 lines
  --corpus FILE    n-gram drafter corpus and unigram baseline text
  --max-calls N    stop after N API calls
  --no-cache       do not read or write the answer cache
  -v, --verbose    log each request
  --version        print the version
  --help           print this help

gen flags
  --words N        token budget, default 30
  --min-words N    masked floor for the <end> option, default 6
  --temp F         sampling temperature, 0 for greedy, default 0.7
  --top-p F        nucleus threshold, default 0.95
  --seed N         seed for this implementation's PRNG, default 0
  --trace          print one line per round trip
  --draft          propose chunks with the n-gram drafter and verify them
  --accept F       absolute Noul floor for a chunk, default 0.25
  --keep F         fraction of the best chunk score a candidate must retain, default 0.8
  --candidates N   chunks proposed per step, default 6
  --order N        drafter n-gram order, default 3
  --chunk-len N    longest drafter chunk in tokens, default 8

eval flags
  --file FILE      held-out text, defaults to the corpus
  --positions N    positions to score, default 40
  --rows N         context rows to print, default 6

Environment
  TYPESAFE_API_KEY        the API key
  TYPESAFE_API_KEY_FILE   file holding the key, default ~/Tokens/TYPESAFE_API_KEY.txt
  XDG_CACHE_HOME          cache location, default ~/.cache/jev-lm
`;

function parseArgs(argv: readonly string[]): Parsed | { error: string } {
  const flags = new Map<string, string | boolean>();
  const positionals: string[] = [];
  let command = "";

  for (let index = 0; index < argv.length; index += 1) {
    const token = argv[index];
    if (token === undefined) continue;

    if (!token.startsWith("-")) {
      if (command === "") command = token;
      else positionals.push(token);
      continue;
    }

    const spec: FlagSpec = command === "" ? GLOBAL_FLAGS : { ...GLOBAL_FLAGS, ...COMMAND_FLAGS[command] };
    const kind = spec[token];
    if (kind === undefined) {
      return { error: `unknown flag ${token}${command === "" ? "" : ` for ${command}`}` };
    }
    if (kind === "boolean") {
      flags.set(token, true);
      continue;
    }
    const value = argv[index + 1];
    if (value === undefined) return { error: `flag ${token} needs a value` };
    flags.set(token, value);
    index += 1;
  }

  return { command, positionals, flags };
}

function numberFlag(parsed: Parsed, name: string, fallback: number): number {
  const raw = parsed.flags.get(name);
  if (raw === undefined || raw === true) return fallback;
  const value = Number(raw);
  if (!Number.isFinite(value)) throw new JevError(`${name} expects a number, got ${raw}`);
  return value;
}

function stringFlag(parsed: Parsed, name: string): string | undefined {
  const raw = parsed.flags.get(name);
  return typeof raw === "string" ? raw : undefined;
}

function booleanFlag(parsed: Parsed, name: string): boolean {
  return parsed.flags.get(name) === true;
}

function buildClient(parsed: Parsed): JevClient {
  return new JevClient({
    model: stringFlag(parsed, "--model") ?? "jev-latest",
    maxCalls: numberFlag(parsed, "--max-calls", 0),
    verbose: booleanFlag(parsed, "--verbose") || booleanFlag(parsed, "-v"),
    cacheDir: booleanFlag(parsed, "--no-cache") ? null : undefined,
  });
}

async function commandProbe(parsed: Parsed): Promise<number> {
  const prompt = parsed.positionals[0];
  if (prompt === undefined) throw new JevError("probe needs a prompt");
  const vocab = Vocab.load(stringFlag(parsed, "--vocab"));
  const client = buildClient(parsed);
  const answers = await client.ask(
    { text_so_far: prompt },
    { next: choice(INSTRUCTION, vocab.options()) },
  );
  const answer = answers["next"];
  if (answer === undefined || answer.type !== "choice") {
    throw new JevError("the next-word answer was missing or not a Choice");
  }
  console.log(`prompt: ${JSON.stringify(prompt)}`);
  console.log(`choice=${JSON.stringify(answer.choice)} confidence=${answer.confidence.toFixed(3)}`);
  const ranked = Object.entries(answer.probabilities).sort(([, a], [, b]) => b - a);
  for (const [token, probability] of ranked.slice(0, numberFlag(parsed, "--top", 12))) {
    console.log(`  ${token.padStart(10)}  ${probability.toFixed(4)}`);
  }
  console.log(`  p(<end>)=${(answer.probabilities["<end>"] ?? 0).toFixed(4)}`);
  console.log(client.usage.toString());
  return 0;
}

async function commandVerify(parsed: Parsed): Promise<number> {
  const [prompt, ...candidates] = parsed.positionals;
  if (prompt === undefined || candidates.length === 0) {
    throw new JevError("verify needs a prompt and at least one candidate chunk");
  }
  const client = buildClient(parsed);
  const questions: Questions = {};
  candidates.forEach((candidate, index) => {
    questions[`chunk_${index}`] = noul(
      `Immediately after \`text_so_far\` comes the text ${quote(candidate)}.`,
    );
  });
  questions["done"] = noul(DONE_INSTRUCTION);
  const answers = await client.ask({ text_so_far: prompt }, questions);
  candidates.forEach((candidate, index) => {
    const answer = answers[`chunk_${index}`];
    const value = answer !== undefined && answer.type === "noul" ? answer.noul : Number.NaN;
    console.log(`  ${value.toFixed(3)}  ${JSON.stringify(candidate)}`);
  });
  const done = answers["done"];
  const doneValue = done !== undefined && done.type === "noul" ? done.noul : Number.NaN;
  console.log(`  ${doneValue.toFixed(3)}  <already complete>`);
  console.log(client.usage.toString());
  return 0;
}

async function commandGen(parsed: Parsed): Promise<number> {
  const prompt = parsed.positionals[0];
  if (prompt === undefined) throw new JevError("gen needs a prompt");
  const vocab = Vocab.load(stringFlag(parsed, "--vocab"));
  const client = buildClient(parsed);
  const drafter = booleanFlag(parsed, "--draft")
    ? Drafter.fromFile(
        stringFlag(parsed, "--corpus") ?? DEFAULT_CORPUS(),
        vocab,
        numberFlag(parsed, "--order", 3),
        numberFlag(parsed, "--chunk-len", 8),
      )
    : null;

  const lm = new WordLM(client, vocab, {
    temp: numberFlag(parsed, "--temp", 0.7),
    topP: numberFlag(parsed, "--top-p", 0.95),
    accept: numberFlag(parsed, "--accept", 0.25),
    keep: numberFlag(parsed, "--keep", 0.8),
    minTokens: numberFlag(parsed, "--min-words", 6),
    maxTokens: numberFlag(parsed, "--words", 30),
    seed: numberFlag(parsed, "--seed", 0),
    candidates: numberFlag(parsed, "--candidates", 6),
    drafter,
  });

  const generation = await lm.generate(prompt);
  if (booleanFlag(parsed, "--trace")) {
    for (const step of generation.steps) console.log(stepLine(step));
    console.log();
  }
  console.log(generation.text);
  console.log();
  console.log(generationReport(generation));
  console.log(client.usage.toString());
  return 0;
}

async function commandEval(parsed: Parsed): Promise<number> {
  const vocab = Vocab.load(stringFlag(parsed, "--vocab"));
  const client = buildClient(parsed);
  const corpus = readFileSync(stringFlag(parsed, "--corpus") ?? DEFAULT_CORPUS(), "utf8");
  const file = stringFlag(parsed, "--file");
  const text = file === undefined ? corpus : readFileSync(file, "utf8");

  const report = await runEval(client, vocab, text, corpus, {
    positions: numberFlag(parsed, "--positions", 40),
    rows: numberFlag(parsed, "--rows", 6),
    verbose: booleanFlag(parsed, "--verbose") || booleanFlag(parsed, "-v"),
  });
  console.log(evalTable(report));
  console.log(client.usage.toString());
  return 0;
}

function version(): string {
  const manifest = readFileSync(new URL("../package.json", import.meta.url), "utf8");
  const parsed = JSON.parse(manifest) as { version?: string };
  return parsed.version ?? "0.0.0";
}

export async function main(argv: readonly string[]): Promise<number> {
  const parsed = parseArgs(argv);
  if ("error" in parsed) {
    console.error(`error: ${parsed.error}`);
    console.error("run jev-lm --help for usage");
    return 1;
  }
  if (booleanFlag(parsed, "--help") || booleanFlag(parsed, "-h")) {
    console.log(USAGE);
    return 0;
  }
  if (booleanFlag(parsed, "--version")) {
    console.log(version());
    return 0;
  }
  if (parsed.command === "") {
    console.error(USAGE);
    return 1;
  }

  try {
    switch (parsed.command) {
      case "probe":
        return await commandProbe(parsed);
      case "verify":
        return await commandVerify(parsed);
      case "gen":
        return await commandGen(parsed);
      case "eval":
        return await commandEval(parsed);
      default:
        console.error(`error: unknown command ${parsed.command}`);
        console.error("run jev-lm --help for usage");
        return 1;
    }
  } catch (error) {
    console.error(`error: ${error instanceof Error ? error.message : String(error)}`);
    return 1;
  }
}

process.exitCode = await main(process.argv.slice(2));
