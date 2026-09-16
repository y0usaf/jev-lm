/**
 * Vocabulary loading, rendering, tokenizing, and the option-masking rules.
 */

import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

export const END = "<end>";
export const END_DESCRIPTION = "nothing should follow; the text is finished";
const PUNCTUATION = new Set([".", ",", "?", "!", ":", ";", "...", "-"]);

const PACKAGE_ROOT = join(dirname(fileURLToPath(import.meta.url)), "..");

/** Resolve a file inside the shipped `data/` directory. */
export function dataPath(...parts: string[]): string {
  return join(PACKAGE_ROOT, "data", ...parts);
}

export const DEFAULT_VOCAB = (): string => dataPath("vocab_en.txt");
export const DEFAULT_CORPUS = (): string => dataPath("corpus_en.txt");
export const DEFAULT_HELDOUT = (): string => dataPath("heldout_en.txt");

export class Vocab {
  readonly tokens: readonly string[];
  private readonly tokenSet: Set<string>;
  private readonly widest: number;

  constructor(tokens: readonly string[]) {
    if (tokens.length > 255) {
      throw new Error(`${tokens.length} options supplied; a Choice accepts at most 255`);
    }
    this.tokens = tokens;
    this.tokenSet = new Set(tokens);
    this.widest = tokens.reduce((widest, token) => Math.max(widest, token.split(" ").length), 1);
  }

  static load(path?: string): Vocab {
    const source = path ?? DEFAULT_VOCAB();
    const tokens = readFileSync(source, "utf8")
      .split("\n")
      .map((line) => line.trim())
      .filter((line) => line !== "" && !line.startsWith("#"));
    try {
      return new Vocab(tokens);
    } catch (error) {
      throw new Error(`${source}: ${error instanceof Error ? error.message : String(error)}`);
    }
  }

  /**
   * The criteria map for a Choice question. Descriptions stay null where the name
   * speaks for itself: every option costs tokens on every call.
   */
  options(): Record<string, string | null> {
    const options: Record<string, string | null> = {};
    for (const token of this.tokens) {
      options[token] = token === END ? END_DESCRIPTION : null;
    }
    return options;
  }

  /** Join tokens into readable text. Punctuation sticks to the previous token. */
  render(tokens: readonly string[]): string {
    let out = "";
    for (const token of tokens) {
      if (out === "") out = token;
      else if (PUNCTUATION.has(token)) out += token;
      else out += ` ${token}`;
    }
    return out;
  }

  /** Longest-match tokenize. null marks a word with no matching option. */
  tokenize(text: string): (string | null)[] {
    const words = text.split(/\s+/).filter((word) => word !== "");
    const tokens: (string | null)[] = [];
    let index = 0;
    while (index < words.length) {
      let matched = false;
      for (let width = this.widest; width >= 1; width -= 1) {
        const candidate = words.slice(index, index + width).join(" ");
        if (candidate === "") continue;
        const canonical = candidate.toLowerCase();
        if (this.tokenSet.has(canonical)) {
          tokens.push(canonical);
          index += width;
          matched = true;
          break;
        }
        if (this.tokenSet.has(candidate)) {
          tokens.push(candidate);
          index += width;
          matched = true;
          break;
        }
      }
      if (!matched) {
        tokens.push(null);
        index += 1;
      }
    }
    return tokens;
  }

  /**
   * Options that survive the anti-degeneracy mask.
   *
   * Measured failure mode: given text ending in a space or a period, Jev puts most of
   * its mass on repeating that character or word. Masking the repeat in code, then
   * renormalizing, is what breaks the loop.
   */
  allowed(tokens: readonly string[], minTokens = 0, banRepeatBigrams = true): Set<string> {
    const allowed = new Set(this.tokens);
    if (tokens.length < minTokens) allowed.delete(END);
    const last = tokens.at(-1);
    if (last !== undefined) allowed.delete(last);
    if (banRepeatBigrams && tokens.length >= 2) {
      const seen = new Set<string>();
      for (let index = 0; index < tokens.length - 1; index += 1) {
        seen.add(`${tokens[index]}\u0000${tokens[index + 1]}`);
      }
      for (const token of allowed) {
        if (last !== undefined && seen.has(`${last}\u0000${token}`)) allowed.delete(token);
      }
    }
    return allowed;
  }
}

/** Drop masked options and rescale. Falls back to the raw distribution if empty. */
export function renormalize(
  probabilities: Record<string, number>,
  allowed?: ReadonlySet<string> | null,
): Record<string, number> {
  if (allowed === undefined || allowed === null) return { ...probabilities };
  const kept: Record<string, number> = {};
  let total = 0;
  for (const [token, probability] of Object.entries(probabilities)) {
    if (allowed.has(token) && probability > 0) {
      kept[token] = probability;
      total += probability;
    }
  }
  if (total <= 0) return { ...probabilities };
  for (const token of Object.keys(kept)) {
    kept[token] = (kept[token] ?? 0) / total;
  }
  return kept;
}
