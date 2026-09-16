/**
 * Word-level generation: one round trip per step, with masking, sampling, and
 * verification.
 *
 * Everything an LLM normally hides lives here: the tokenizer, the sampler, the
 * repetition mask, the stop rule, and a verified-chunk path instead of a cache.
 */

import { JevClient, choice, noul, quote } from "./api.js";
import { Drafter, countWords, verifiedChunk } from "./draft.js";
import { END, Vocab, renormalize } from "./vocab.js";

export const INSTRUCTION = "Which word comes next in `text_so_far`?";
export const DONE_INSTRUCTION =
  "`text_so_far` is already a complete and natural sentence as it stands.";

/** Seeded PRNG, so a seed reproduces a run within this implementation. */
function mulberry32(seed: number): () => number {
  let state = seed >>> 0;
  return () => {
    state = (state + 0x6d2b79f5) >>> 0;
    let t = state;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export interface Step {
  index: number;
  seconds: number;
  picked: string;
  pickedP: number;
  endP: number;
  done: number;
  top: [string, number][];
  chunk: string;
  chunkScore: number;
  chunkScores: readonly (readonly [string, number])[];
  masked: number;
  added: number;
  endIgnored: boolean;
  stopped: string;
}

export function stepLine(step: Step): string {
  let head =
    `${String(step.index).padStart(3)}  ${step.picked.padStart(10)}  p=${step.pickedP.toFixed(3)}  ` +
    `end=${step.endP.toFixed(2)}  done=${step.done.toFixed(2)}`;
  if (step.chunk !== "") head += `  chunk=${JSON.stringify(step.chunk)} at ${step.chunkScore.toFixed(2)}`;
  else if (step.chunkScores.length > 0) head += `  best-chunk=${(step.chunkScores[0]?.[1] ?? 0).toFixed(2)}`;
  if (step.masked > 0) head += `  masked=${step.masked}`;
  if (step.endIgnored) head += "  end-ignored";
  if (step.stopped !== "") head += `  stop=${step.stopped}`;
  return `${head}  ${step.seconds.toFixed(2)}s`;
}

export interface Generation {
  prompt: string;
  tokens: string[];
  text: string;
  steps: Step[];
  stopReason: string;
}

export function generationReport(generation: Generation): string {
  const roundTrips = Math.max(generation.steps.length, 1);
  return (
    `tokens=${generation.tokens.length} round-trips=${generation.steps.length} ` +
    `tokens/round-trip=${(generation.tokens.length / roundTrips).toFixed(2)} ` +
    `stop=${generation.stopReason}`
  );
}

export interface WordLMOptions {
  temp?: number;
  topP?: number;
  accept?: number;
  keep?: number;
  stopDone?: number;
  stopEnd?: number;
  stopAgree?: number;
  minNew?: number;
  maxTokens?: number;
  minTokens?: number;
  seed?: number;
  drafter?: Drafter | null;
  candidates?: number;
}

export class WordLM {
  private readonly client: JevClient;
  private readonly vocab: Vocab;
  private readonly temp: number;
  private readonly topP: number;
  private readonly accept: number;
  private readonly keep: number;
  private readonly stopDone: number;
  private readonly stopEnd: number;
  private readonly stopAgree: number;
  private readonly minNew: number;
  private readonly maxTokens: number;
  private readonly minTokens: number;
  private readonly drafter: Drafter | null;
  private readonly candidates: number;
  private readonly random: () => number;

  constructor(client: JevClient, vocab: Vocab, options: WordLMOptions = {}) {
    this.client = client;
    this.vocab = vocab;
    this.temp = options.temp ?? 0.7;
    this.topP = options.topP ?? 0.95;
    this.accept = options.accept ?? 0.25;
    this.keep = options.keep ?? 0.8;
    this.stopDone = options.stopDone ?? 0.5;
    this.stopEnd = options.stopEnd ?? 0.2;
    this.stopAgree = options.stopAgree ?? 0.25;
    this.minNew = options.minNew ?? 1;
    this.maxTokens = options.maxTokens ?? 40;
    this.minTokens = options.minTokens ?? 6;
    this.drafter = options.drafter ?? null;
    this.candidates = options.candidates ?? 6;
    this.random = mulberry32(options.seed ?? 0);
  }

  sample(probabilities: Record<string, number>): [string, number] {
    const entries = Object.entries(probabilities);
    if (this.temp <= 0) {
      let best: [string, number] | null = null;
      for (const entry of entries) if (best === null || entry[1] > best[1]) best = entry;
      if (best === null) throw new Error("empty distribution");
      return best;
    }

    const scaled = entries
      .map(([token, value]) => [token, value ** (1 / this.temp)] as [string, number])
      .sort(([, a], [, b]) => b - a);
    const total = scaled.reduce((sum, [, value]) => sum + value, 0);
    const kept: [string, number][] = [];
    let cumulative = 0;
    for (const [token, value] of scaled) {
      const weight = value / total;
      kept.push([token, weight]);
      cumulative += weight;
      if (cumulative >= this.topP) break;
    }
    const mass = kept.reduce((sum, [, weight]) => sum + weight, 0);
    let roll = this.random() * mass;
    for (const [token, weight] of kept) {
      roll -= weight;
      if (roll <= 0) return [token, probabilities[token] ?? 0];
    }
    const fallback = kept.at(-1);
    if (fallback === undefined) throw new Error("empty distribution after top-p");
    return [fallback[0], probabilities[fallback[0]] ?? 0];
  }

  async step(tokens: readonly string[]): Promise<Step> {
    const text = this.vocab.render(tokens);
    const allowed = this.vocab.allowed(tokens, this.minTokens);
    const proposed = this.drafter === null ? [] : this.drafter.candidates(tokens, this.candidates);

    const questions = {
      next: choice(INSTRUCTION, this.vocab.options()),
      done: noul(DONE_INSTRUCTION),
    } as Record<string, ReturnType<typeof choice> | ReturnType<typeof noul>>;
    proposed.forEach((candidate, index) => {
      questions[`chunk_${index}`] = noul(
        `Immediately after \`text_so_far\` comes the text ${quote(candidate)}.`,
      );
    });

    const started = Date.now();
    const answers = await this.client.ask({ text_so_far: text }, questions);
    const seconds = (Date.now() - started) / 1000;

    const nextAnswer = answers["next"];
    if (nextAnswer === undefined || nextAnswer.type !== "choice") {
      throw new Error("the next-word answer was missing or not a Choice");
    }
    const doneAnswer = answers["done"];
    if (doneAnswer === undefined || doneAnswer.type !== "noul") {
      throw new Error("the done answer was missing or not a Noul");
    }

    const raw = nextAnswer.probabilities;
    const kept = renormalize(raw, allowed);
    const masked = Object.keys(raw).length - Object.keys(kept).length;
    let [picked, pickedP] = this.sample(kept);
    const top = Object.entries(kept)
      .sort(([, a], [, b]) => b - a)
      .slice(0, 5) as [string, number][];
    const done = doneAnswer.noul;
    const endP = kept[END] ?? raw[END] ?? 0;
    const { chunk: accepted, scored } = verifiedChunk(answers, proposed, this.accept, this.keep);
    const chunk = countWords(accepted) >= 2 ? accepted : "";

    // The <end> option maxed at 0.21 over 24 character steps, so it fires on noise, and
    // it must never reach the output text. Substitute the best real token and let the
    // caller's stop rules use endP and done together.
    let endIgnored = false;
    if (picked === END) {
      const fallback = top.find(([token]) => token !== END);
      if (fallback !== undefined) {
        [picked, pickedP] = fallback;
        endIgnored = true;
      }
    }

    return {
      index: tokens.length,
      seconds,
      picked,
      pickedP,
      endP,
      done,
      top,
      chunk,
      chunkScore: scored.find(([candidate]) => candidate === chunk)?.[1] ?? 0,
      chunkScores: scored,
      masked,
      added: 1,
      endIgnored,
      stopped: "",
    };
  }

  async generate(prompt: string): Promise<Generation> {
    const tokens = this.vocab.tokenize(prompt).filter((token): token is string => token !== null);
    const promptLength = tokens.length;
    const steps: Step[] = [];
    let stopReason = "max_tokens";

    while (tokens.length < this.maxTokens) {
      const step = await this.step(tokens);
      steps.push(step);

      // The done and <end> questions were asked about the text before this step's token,
      // so the stop check has to happen before the token is appended.
      const added = tokens.length - promptLength;
      if (step.done >= this.stopDone && added >= this.minNew) {
        step.stopped = "done";
        stopReason = "done";
        break;
      }
      if (step.endP >= this.stopEnd && step.done >= this.stopAgree && added >= this.minNew) {
        step.stopped = "end-option";
        stopReason = "end-option";
        break;
      }

      if (step.chunk !== "") tokens.push(...step.chunk.split(" "));
      else tokens.push(step.picked);
      step.added = tokens.length - step.index;
    }

    return {
      prompt,
      tokens,
      text: this.vocab.render(tokens),
      steps,
      stopReason,
    };
  }
}
