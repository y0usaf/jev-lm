/**
 * N-gram drafter. Proposes whole chunks that a Noul can accept or reject.
 *
 * The measured reason this exists: one round trip costs the same whether you ask one
 * question or six, because questions inside a request run in parallel. So a free local
 * drafter plus one verification request per accepted chunk buys several tokens per
 * round trip instead of one.
 */

import { readFileSync } from "node:fs";

import type { Answers } from "./api.js";
import { END, Vocab } from "./vocab.js";

const PUNCTUATION = new Set([".", ",", "?", "!", ":", ";", "...", "-"]);

export class Drafter {
  readonly vocab: Vocab;
  readonly order: number;
  readonly maxLen: number;
  private readonly counts = new Map<string, Map<string, number>>();

  constructor(vocab: Vocab, order = 3, maxLen = 8) {
    this.vocab = vocab;
    this.order = order;
    this.maxLen = maxLen;
  }

  static fromText(text: string, vocab: Vocab, order = 3, maxLen = 8): Drafter {
    const drafter = new Drafter(vocab, order, maxLen);
    drafter.add(text);
    return drafter;
  }

  /** One document per line, so proposed chunks never cross a sentence break. */
  static fromFile(path: string, vocab: Vocab, order = 3, maxLen = 8): Drafter {
    const drafter = new Drafter(vocab, order, maxLen);
    for (const line of readFileSync(path, "utf8").split("\n")) {
      if (line.trim() !== "" && !line.trimStart().startsWith("#")) drafter.add(line);
    }
    return drafter;
  }

  add(text: string): void {
    const tokens = this.vocab.tokenize(text).filter((token): token is string => token !== null);
    for (let width = 1; width <= this.order; width += 1) {
      const keyWidth = width - 1;
      for (let index = 0; index + width <= tokens.length; index += 1) {
        const key = tokens.slice(index, index + keyWidth).join("\u0000");
        const next = tokens[index + keyWidth];
        if (next === undefined) continue;
        const counter = this.counts.get(key) ?? new Map<string, number>();
        counter.set(next, (counter.get(next) ?? 0) + 1);
        this.counts.set(key, counter);
      }
    }
  }

  /**
   * Longest-match backoff: try the full context, then drop the oldest token.
   *
   * `minWidth` stops the backoff before the empty context. Chunk extension uses
   * minWidth 1 so a chain cannot jump to an unrelated sentence.
   */
  nextToken(context: readonly string[], minWidth = 0): string | null {
    const widest = Math.min(context.length, this.order - 1);
    for (let width = widest; width >= minWidth; width -= 1) {
      const key = width === 0 ? "" : context.slice(context.length - width).join("\u0000");
      const counter = this.counts.get(key);
      if (counter !== undefined && counter.size > 0) return topToken(counter);
    }
    return null;
  }

  draw(context: readonly string[], limit = 2): string[] {
    const widest = Math.min(context.length, this.order - 1);
    for (let width = widest; width >= 0; width -= 1) {
      const key = width === 0 ? "" : context.slice(context.length - width).join("\u0000");
      const counter = this.counts.get(key);
      if (counter !== undefined && counter.size > 0) return topTokens(counter, limit);
    }
    return [];
  }

  /** Whole continuations to verify, longest first, up to k of them. */
  candidates(tokens: readonly string[], k = 6): string[] {
    const context = [...tokens];
    const chains: string[][] = [];
    for (const first of this.draw(context, 2)) {
      const chain = [first];
      chains.push([...chain]);
      for (let step = 0; step < this.maxLen - 1; step += 1) {
        const following = this.nextToken([...context, ...chain], 1);
        if (following === null || following === END) break;
        chain.push(following);
        chains.push([...chain]);
      }
    }

    const rendered: string[] = [];
    for (const chain of chains.sort((a, b) => b.length - a.length)) {
      const text = chain.join(" ");
      if (!rendered.includes(text)) rendered.push(text);
    }
    return rendered.slice(0, k);
  }
}

function topTokens(counter: ReadonlyMap<string, number>, limit: number): string[] {
  return [...counter.entries()]
    .sort(([, a], [, b]) => b - a)
    .slice(0, limit)
    .map(([token]) => token);
}

function topToken(counter: ReadonlyMap<string, number>): string | null {
  return topTokens(counter, 1)[0] ?? null;
}

export interface ChunkChoice {
  readonly chunk: string;
  readonly scored: readonly (readonly [string, number])[];
}

/**
 * Read the chunk Nouls out of one request and return the chunk to insert.
 *
 * Rule: keep the candidates scoring at least `keep` times the best score, above an
 * absolute `accept` floor, then take the longest survivor. Measured behavior:
 *
 * - Unrelated candidates separate well: `paris` 0.95 against `london` 0.02.
 * - Nested candidates barely separate, and the shortest wins slightly: `and` 0.44,
 *   `and looked` 0.42, `and looked at the street` 0.38. A length-agnostic rule picks one
 *   token; keep 0.8 picks the whole chain, which is the point of the drafter.
 * - Four invented continuations all landed at 0.14 to 0.17, below any sane floor.
 */
export function verifiedChunk(
  answers: Answers,
  candidates: readonly string[],
  accept = 0.25,
  keep = 0.8,
): ChunkChoice {
  const scored: [string, number][] = [];
  candidates.forEach((candidate, index) => {
    const answer = answers[`chunk_${index}`];
    if (answer !== undefined && answer.type === "noul") scored.push([candidate, answer.noul]);
  });
  if (scored.length === 0) return { chunk: "", scored };

  const top = Math.max(...scored.map(([, value]) => value));
  const threshold = Math.max(accept, top * keep);
  const qualified = scored.filter(([, value]) => value >= threshold);
  const ranked = [...scored].sort(([, a], [, b]) => b - a);
  if (qualified.length === 0) return { chunk: "", scored: ranked };

  let best = qualified[0];
  if (best === undefined) return { chunk: "", scored: ranked };
  for (const entry of qualified) {
    const [candidate, value] = entry;
    const [bestCandidate, bestValue] = best;
    if (
      countWords(candidate) > countWords(bestCandidate) ||
      (countWords(candidate) === countWords(bestCandidate) && value > bestValue)
    ) {
      best = entry;
    }
  }
  return { chunk: best[0], scored: ranked };
}

export function countWords(candidate: string): number {
  const words = candidate.split(" ").filter((word) => !PUNCTUATION.has(word));
  return (words.length > 0 ? words : candidate.split(" ")).length;
}
