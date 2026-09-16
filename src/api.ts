/**
 * HTTP client for the TypeSafe System One endpoint.
 *
 * One POST per round trip. Timing and token use accumulate on `usage`.
 */

import { createHash } from "node:crypto";
import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { homedir } from "node:os";
import { join } from "node:path";

export const ENDPOINT = "https://api.typesafe.ai/v1/systemone";
export const DEFAULT_MODEL = "jev-latest";
export const DEFAULT_KEY_PATH = "~/Tokens/TYPESAFE_API_KEY.txt";

const RETRY_STATUS = new Set([408, 409, 429, 500, 502, 503, 504, 529]);

export type JsonValue =
  | string
  | number
  | boolean
  | null
  | readonly JsonValue[]
  | { readonly [key: string]: JsonValue };

export interface NoulQuestion {
  readonly type: "noul";
  readonly instructions: string;
  readonly criteria?: Record<string, string>;
}

export interface ChoiceQuestion {
  readonly type: "choice";
  readonly instructions: string;
  readonly criteria: Record<string, string | null>;
}

export type Question = NoulQuestion | ChoiceQuestion;
export type Questions = Record<string, Question>;

export interface NoulAnswer {
  readonly type: "noul";
  readonly noul: number;
}

export interface ChoiceAnswer {
  readonly type: "choice";
  readonly choice: string;
  readonly probabilities: Record<string, number>;
  readonly confidence: number;
}

export type Answer = NoulAnswer | ChoiceAnswer;
export type Answers = Record<string, Answer>;

export class JevError extends Error {
  override readonly name: string = "JevError";
}

/** An HTTP failure, carrying whether the status is worth retrying. */
class HttpError extends JevError {
  override readonly name: string = "HttpError";
  readonly retryable: boolean;

  constructor(status: number, detail: string) {
    super(`HTTP ${status} from ${ENDPOINT}: ${detail}`);
    this.retryable = RETRY_STATUS.has(status);
  }
}

/** Resolve the key from an argument, the environment, then a file. */
export function resolveApiKey(explicit?: string, path?: string): string {
  if (explicit !== undefined && explicit.trim() !== "") return explicit.trim();
  const fromEnv = process.env["TYPESAFE_API_KEY"];
  if (fromEnv !== undefined && fromEnv.trim() !== "") return fromEnv.trim();
  const candidate = expandHome(path ?? process.env["TYPESAFE_API_KEY_FILE"] ?? DEFAULT_KEY_PATH);
  if (existsSync(candidate)) {
    const key = readFileSync(candidate, "utf8").trim();
    if (key !== "") return key;
  }
  throw new JevError(
    `no API key found: pass one, set TYPESAFE_API_KEY, or write one to ${candidate}`,
  );
}

export function expandHome(path: string): string {
  return path.startsWith("~/") ? join(homedir(), path.slice(2)) : path;
}

export class Usage {
  calls = 0;
  inputTokens = 0;
  outputTokens = 0;
  seconds = 0;
  cached = 0;

  add(inputTokens: number, outputTokens: number, seconds: number): void {
    this.calls += 1;
    this.inputTokens += inputTokens;
    this.outputTokens += outputTokens;
    this.seconds += seconds;
  }

  toString(): string {
    const cached = this.cached > 0 ? `, ${this.cached} cached` : "";
    return `${this.calls} calls, ${this.inputTokens} in, ${this.outputTokens} out, ${this.seconds.toFixed(1)}s${cached}`;
  }
}

export interface ClientOptions {
  key?: string;
  keyPath?: string;
  model?: string;
  timeoutMs?: number;
  retries?: number;
  maxCalls?: number;
  cacheDir?: string | null;
  verbose?: boolean;
}

/** Deterministic key order so the same request always hashes to the same cache file. */
function stableStringify(value: unknown): string {
  if (value === null || typeof value !== "object") return JSON.stringify(value) ?? "null";
  if (Array.isArray(value)) return `[${value.map(stableStringify).join(",")}]`;
  const entries = Object.entries(value as Record<string, unknown>)
    .filter(([, item]) => item !== undefined)
    .sort(([a], [b]) => (a < b ? -1 : a > b ? 1 : 0));
  return `{${entries.map(([key, item]) => `${JSON.stringify(key)}:${stableStringify(item)}`).join(",")}}`;
}

function digest(state: JsonValue, questions: Questions, model: string): string {
  const payload = stableStringify({ state, questions, model });
  return createHash("sha256").update(payload).digest("hex").slice(0, 24);
}

export class JevClient {
  readonly usage = new Usage();
  readonly model: string;
  private readonly key: string;
  private readonly timeoutMs: number;
  private readonly retries: number;
  private readonly maxCalls: number;
  private readonly cacheDir: string | null;
  private readonly verbose: boolean;

  constructor(options: ClientOptions = {}) {
    this.key = resolveApiKey(options.key, options.keyPath);
    this.model = options.model ?? DEFAULT_MODEL;
    this.timeoutMs = options.timeoutMs ?? 120_000;
    this.retries = options.retries ?? 4;
    this.maxCalls = options.maxCalls ?? 0;
    this.verbose = options.verbose ?? false;
    const cacheDir = options.cacheDir === undefined ? "~/.cache/jev-cli" : options.cacheDir;
    this.cacheDir = cacheDir === null ? null : expandHome(cacheDir);
    if (this.cacheDir !== null) mkdirSync(this.cacheDir, { recursive: true });
  }

  async ask(state: JsonValue, questions: Questions): Promise<Answers> {
    if (this.maxCalls > 0 && this.usage.calls >= this.maxCalls) {
      throw new JevError(`call budget of ${this.maxCalls} reached`);
    }

    const cachePath =
      this.cacheDir === null ? null : join(this.cacheDir, `${digest(state, questions, this.model)}.json`);
    if (cachePath !== null && existsSync(cachePath)) {
      const cached = JSON.parse(readFileSync(cachePath, "utf8")) as { answers: Answers };
      this.usage.cached += 1;
      return cached.answers;
    }

    const body = JSON.stringify({ state, model: this.model, questions });
    let lastError: Error | null = null;

    for (let attempt = 0; attempt <= this.retries; attempt += 1) {
      const started = Date.now();
      try {
        const response = await fetch(ENDPOINT, {
          method: "POST",
          headers: {
            Authorization: `Bearer ${this.key}`,
            "Content-Type": "application/json",
          },
          body,
          signal: AbortSignal.timeout(this.timeoutMs),
        });
        if (!response.ok) {
          throw new HttpError(response.status, (await response.text()).slice(0, 400));
        }
        const payload = (await response.json()) as {
          answers: Answers;
          usage?: { input_tokens?: number; output_tokens?: number };
        };
        const seconds = (Date.now() - started) / 1000;
        this.usage.add(payload.usage?.input_tokens ?? 0, payload.usage?.output_tokens ?? 0, seconds);
        if (this.verbose) {
          console.log(
            `  [jev] ${Object.keys(questions).length} questions, ${seconds.toFixed(2)}s, ` +
              `${payload.usage?.input_tokens ?? 0} in, ${payload.usage?.output_tokens ?? 0} out`,
          );
        }
        if (cachePath !== null) {
          try {
            writeFileSync(cachePath, JSON.stringify({ answers: payload.answers }));
          } catch {
            // A read-only or full cache directory must not fail the request.
          }
        }
        return payload.answers;
      } catch (error) {
        if (error instanceof HttpError && !error.retryable) throw error;
        lastError = error instanceof Error ? error : new JevError(String(error));
        if (attempt < this.retries) {
          await new Promise((resolve) => setTimeout(resolve, Math.min(2000 * 2 ** attempt, 20_000)));
        }
      }
    }

    throw lastError ?? new JevError("request failed");
  }
}

export function noul(instructions: string, criteria?: Record<string, string>): NoulQuestion {
  return criteria === undefined ? { type: "noul", instructions } : { type: "noul", instructions, criteria };
}

export function choice(instructions: string, options: Record<string, string | null>): ChoiceQuestion {
  return { type: "choice", instructions, criteria: options };
}

/** Single-quote a chunk for embedding in a Noul instruction. */
export function quote(text: string): string {
  return `'${text.trim().replace(/\s+/g, " ").replace(/'/g, "\\'")}'`;
}
