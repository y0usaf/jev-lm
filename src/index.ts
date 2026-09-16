/**
 * Jev-LM: a word-level language model whose output layer is Jev.
 *
 * ```ts
 * import { JevClient, Vocab, WordLM, Drafter } from "jev-lm";
 *
 * const client = new JevClient();
 * const vocab = Vocab.load();
 * const lm = new WordLM(client, vocab, { maxTokens: 20 });
 * const generation = await lm.generate("the little dog");
 * console.log(generation.text);
 * ```
 */

export {
  DEFAULT_KEY_PATH,
  DEFAULT_MODEL,
  ENDPOINT,
  JevClient,
  JevError,
  Usage,
  choice,
  expandHome,
  noul,
  quote,
  resolveApiKey,
} from "./api.js";
export type {
  Answer,
  Answers,
  ChoiceAnswer,
  ChoiceQuestion,
  ClientOptions,
  JsonValue,
  NoulAnswer,
  NoulQuestion,
  Question,
  Questions,
} from "./api.js";

export { DEFAULT_CORPUS, DEFAULT_HELDOUT, DEFAULT_VOCAB, END, Vocab, dataPath, renormalize } from "./vocab.js";

export { Drafter, countWords, verifiedChunk } from "./draft.js";
export type { ChunkChoice } from "./draft.js";

export { DONE_INSTRUCTION, INSTRUCTION, WordLM, generationReport, stepLine } from "./lm.js";
export type { Generation, Step, WordLMOptions } from "./lm.js";

export { FLOOR, bitsPerToken, evalTable, runEval, unigramPerToken, uniformPerToken } from "./eval.js";
export type { EvalOptions, EvalReport } from "./eval.js";
