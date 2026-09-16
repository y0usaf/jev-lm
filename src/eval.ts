/**
 * Bits-per-token measurement against a unigram baseline over the same vocabulary.
 *
 * This is the check that decides whether the contraption is a language model at all.
 * Measured at character granularity, Jev scored 4.19 bits/char against 4.00 for a
 * unigram model trained on 400 characters, so the context bought nothing. At word
 * granularity it scores 6.82 against a unigram at 6.18 on the shipped held-out file.
 */

import { JevClient, choice } from "./api.js";
import { INSTRUCTION } from "./lm.js";
import { Vocab } from "./vocab.js";

/** Probabilities come back quantized at about 1 percent, so a lower floor is meaningless. */
export const FLOOR = 1e-4;

export interface EvalReport {
  scored: number;
  skipped: number;
  jevBits: number;
  unigramBits: number;
  uniformBits: number;
  rows: [string, string, number, number][];
}

export function bitsPerToken(report: EvalReport): number {
  return report.scored > 0 ? report.jevBits / report.scored : 0;
}

export function unigramPerToken(report: EvalReport): number {
  return report.scored > 0 ? report.unigramBits / report.scored : 0;
}

export function uniformPerToken(report: EvalReport): number {
  return report.scored > 0 ? report.uniformBits / report.scored : 0;
}

export function evalTable(report: EvalReport): string {
  const lines = [`${"context".padEnd(44)} ${"actual".padStart(10)} ${"jev p".padStart(8)} ${"unigram p".padStart(10)}`];
  for (const [context, actual, jevP, unigramP] of report.rows) {
    lines.push(
      `...${context.slice(-39).padEnd(41)} ${actual.padStart(10)} ${jevP.toFixed(4).padStart(8)} ${unigramP.toFixed(4).padStart(10)}`,
    );
  }
  lines.push("");
  lines.push(`scored ${report.scored} positions, skipped ${report.skipped} out-of-vocabulary`);
  lines.push(`  Jev      ${report.jevBits.toFixed(2).padStart(8)} bits, ${bitsPerToken(report).toFixed(2)} bits/token`);
  lines.push(`  unigram  ${report.unigramBits.toFixed(2).padStart(8)} bits, ${unigramPerToken(report).toFixed(2)} bits/token`);
  lines.push(`  uniform  ${report.uniformBits.toFixed(2).padStart(8)} bits, ${uniformPerToken(report).toFixed(2)} bits/token`);
  const verdict =
    bitsPerToken(report) < unigramPerToken(report) - 0.05
      ? "context is carrying information"
      : "context is NOT beating a unigram table";
  lines.push(`  verdict: ${verdict}`);
  return lines.join("\n");
}

export interface EvalOptions {
  positions?: number;
  rows?: number;
  verbose?: boolean;
}

export async function runEval(
  client: JevClient,
  vocab: Vocab,
  text: string,
  corpusText: string,
  options: EvalOptions = {},
): Promise<EvalReport> {
  const positions = options.positions ?? 40;
  const rowCount = options.rows ?? 6;
  const tokens = vocab.tokenize(text);

  const corpus = new Map<string, number>();
  for (const token of vocab.tokenize(corpusText)) {
    if (token !== null) corpus.set(token, (corpus.get(token) ?? 0) + 1);
  }
  const corpusTotal = [...corpus.values()].reduce((sum, count) => sum + count, 0);
  const uniform = Math.log2(vocab.tokens.length);

  const report: EvalReport = {
    scored: 0,
    skipped: 0,
    jevBits: 0,
    unigramBits: 0,
    uniformBits: 0,
    rows: [],
  };

  const limit = Math.min(tokens.length, positions + 1);
  for (let index = 1; index < limit; index += 1) {
    const actual = tokens[index];
    if (actual === undefined || actual === null) {
      report.skipped += 1;
      continue;
    }
    const prefix = tokens.slice(0, index).filter((token): token is string => token !== null);
    const answers = await client.ask(
      { text_so_far: vocab.render(prefix) },
      { next: choice(INSTRUCTION, vocab.options()) },
    );
    const answer = answers["next"];
    if (answer === undefined || answer.type !== "choice") {
      throw new Error("the next-word answer was missing or not a Choice");
    }
    const jevP = Math.max(answer.probabilities[actual] ?? 0, FLOOR);
    const unigramP = ((corpus.get(actual) ?? 0) + 0.5) / (corpusTotal + 0.5 * vocab.tokens.length);
    report.jevBits += -Math.log2(jevP);
    report.unigramBits += -Math.log2(unigramP);
    report.uniformBits += uniform;
    report.scored += 1;
    if (report.rows.length < rowCount) {
      report.rows.push([vocab.render(prefix), actual, jevP, unigramP]);
    }
    if (options.verbose === true) {
      console.log(
        `  ${String(report.scored).padStart(4)}  ${actual.padStart(10)}  jev p=${jevP.toFixed(4)}  unigram p=${unigramP.toFixed(4)}`,
      );
    }
  }
  return report;
}
