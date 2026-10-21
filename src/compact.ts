import { noulAnswer } from './request.js';
import {
  DEFAULT_MAX_CANDIDATE_CALLS,
  buildGroups,
  collectToolCalls,
  estimateTokens,
  fitState,
  selectCandidates,
} from './state.js';
import type {
  CallAnswer,
  CallDecision,
  CallGroup,
  CompactOptions,
  CompactResult,
  CompactionState,
  JevAsker,
  JevQuestions,
  Message,
  ResolvedCompactOptions,
  ToolCall,
  ToolUse,
} from './types.js';

export const DEFAULT_OPTIONS: ResolvedCompactOptions = {
  goal: '',
  keepThreshold: 0.5,
  preserveRecentMessages: 6,
  maxStateTokens: 25_000,
  maxStateChars: 40_000,
  maxRequestTokens: 30_000,
  maxCandidateCalls: DEFAULT_MAX_CANDIDATE_CALLS,
  truncateHeadChars: 300,
};

/** Tokens the request envelope (`model`, key names) adds around state and questions. */
const REQUEST_OVERHEAD_TOKENS = 20;

/** Items (calls or groups) per request: two `noul` questions each, so a batch
 *  stays inside laya-serve's MAX_QUESTIONS of 64 per request. */
const MAX_ITEMS_PER_BATCH = 32;

function finite(value: number | undefined, fallback: number): number {
  return typeof value === 'number' && Number.isFinite(value) ? value : fallback;
}

export function resolveOptions(options: CompactOptions = {}): ResolvedCompactOptions {
  return {
    goal: options.goal ?? DEFAULT_OPTIONS.goal,
    keepThreshold: finite(options.keepThreshold, DEFAULT_OPTIONS.keepThreshold),
    preserveRecentMessages: Math.max(
      0,
      Math.floor(
        finite(options.preserveRecentMessages, DEFAULT_OPTIONS.preserveRecentMessages),
      ),
    ),
    maxStateTokens: Math.max(1, finite(options.maxStateTokens, DEFAULT_OPTIONS.maxStateTokens)),
    maxStateChars: Math.max(1, finite(options.maxStateChars, DEFAULT_OPTIONS.maxStateChars)),
    maxRequestTokens: Math.max(
      1,
      finite(options.maxRequestTokens, DEFAULT_OPTIONS.maxRequestTokens),
    ),
    maxCandidateCalls: Math.max(
      0,
      Math.floor(finite(options.maxCandidateCalls, DEFAULT_OPTIONS.maxCandidateCalls)),
    ),
    truncateHeadChars: Math.max(
      0,
      Math.floor(finite(options.truncateHeadChars, DEFAULT_OPTIONS.truncateHeadChars)),
    ),
  };
}

/** The two `noul` questions asked about one call: keep the call, keep its result. */
export function questionsFor(call: ToolCall): JevQuestions {
  return {
    [`call_${call.id}`]: {
      type: 'noul',
      instructions: `Tool call ${call.id} (${call.tool}) should stay in the history: knowing this call was made, with its input, still matters for what the assistant does next`,
    },
    [`result_${call.id}`]: {
      type: 'noul',
      instructions: `The full output of tool call ${call.id} (${call.tool}, ${call.resultChars} chars) should stay in the history verbatim: the assistant still needs its contents and re-running the tool would not do`,
    },
  };
}

/** The two `noul` questions asked about one group of old calls, as a whole. */
export function groupQuestions(group: CallGroup): JevQuestions {
  return {
    [`call_${group.id}`]: {
      type: 'noul',
      instructions: `Group ${group.id} covers ${group.count} earlier ${group.tool} tool calls: these calls should stay in the history as call entries, since knowing they were made, with their inputs, still matters for what the assistant does next`,
    },
    [`result_${group.id}`]: {
      type: 'noul',
      instructions: `Group ${group.id}: the full outputs of those ${group.count} earlier ${group.tool} tool calls should stay in the history verbatim: the assistant still needs their contents and re-running the tools would not do`,
    },
  };
}

/**
 * Splits the items into batches whose questions, together with the (always
 * complete) state, fit one request: by token budget and never more than
 * `MAX_ITEMS_PER_BATCH` items (two questions each) per batch.
 */
export function batchCalls<T extends { id: string }>(
  items: readonly T[],
  stateTokens: number,
  options: Pick<ResolvedCompactOptions, 'maxRequestTokens'>,
  questionsOf: (item: T) => JevQuestions = (item) => questionsFor(item as unknown as ToolCall),
): T[][] {
  const budget = options.maxRequestTokens - stateTokens - REQUEST_OVERHEAD_TOKENS;
  const batches: T[][] = [];
  let current: T[] = [];
  let currentTokens = 0;
  for (const item of items) {
    const tokens = estimateTokens(JSON.stringify(questionsOf(item)));
    if (
      current.length > 0 &&
      (current.length >= MAX_ITEMS_PER_BATCH || currentTokens + tokens > budget)
    ) {
      batches.push(current);
      current = [];
      currentTokens = 0;
    }
    if (current.length === 0 && tokens > budget) {
      throw new Error(
        `state leaves no room for questions (~${stateTokens} of ${options.maxRequestTokens} tokens)`,
      );
    }
    current.push(item);
    currentTokens += tokens;
  }
  if (current.length > 0) batches.push(current);
  return batches;
}

export function decideCall(
  call: Pick<ToolCall, 'id' | 'tool' | 'pinned'>,
  answer: CallAnswer,
  options: Pick<ResolvedCompactOptions, 'keepThreshold'>,
): CallDecision {
  const base = { id: call.id, tool: call.tool, ...answer };
  if (call.pinned) return { ...base, action: 'keep', reason: 'pinned' };
  if (answer.keepResult >= options.keepThreshold) {
    return { ...base, action: 'keep', reason: 'kept' };
  }
  if (answer.keepCall >= options.keepThreshold) {
    return { ...base, action: 'drop_result', reason: 'result_dropped' };
  }
  return { ...base, action: 'drop_call', reason: 'call_dropped' };
}

async function askBatch<T extends { id: string }>(
  asker: JevAsker,
  state: CompactionState,
  batch: readonly T[],
  questionsOf: (item: T) => JevQuestions,
): Promise<Map<string, CallAnswer>> {
  const questions: JevQuestions = Object.assign({}, ...batch.map(questionsOf));
  const { answers } = await asker.ask(state, questions);
  return new Map(
    batch.map((item) => [
      item.id,
      {
        keepCall: noulAnswer(answers, `call_${item.id}`),
        keepResult: noulAnswer(answers, `result_${item.id}`),
      },
    ]),
  );
}

function truncatedResultText(text: string, isError: boolean, headChars: number): string {
  if (text.length <= headChars + 120) return text;
  const head = headChars > 0 ? `${text.slice(0, headChars)}\n` : '';
  return `${head}[fast-jev-compaction truncated ${text.length - headChars} chars of this tool result${
    isError ? ' (error)' : ''
  }; re-run the tool if needed]`;
}

/**
 * Rebuilds the conversation from the decisions. A dropped call disappears
 * together with its result; a dropped result keeps a bounded head and note.
 * Messages that lose all their content are removed; untouched messages are
 * returned as the same objects they came in as.
 */
export function applyDecisions(
  messages: readonly Message[],
  decisions: readonly CallDecision[],
  calls: readonly ToolCall[],
  headChars: number,
): Message[] {
  const byId = new Map(calls.map((call) => [call.id, call]));
  const actions = new Map<string, CallDecision['action']>();
  for (const decision of decisions) {
    const call = byId.get(decision.id);
    if (call && decision.action !== 'keep') actions.set(call.tool_use_id, decision.action);
  }
  const kept: Message[] = [];
  for (const message of messages) {
    const touched =
      message.toolUses.some((tool) => actions.has(tool.tool_use_id)) ||
      (message.toolResults ?? []).some((result) => actions.has(result.tool_use_id));
    if (!touched) {
      kept.push(message);
      continue;
    }
    const toolUses = message.toolUses
      .filter((tool) => actions.get(tool.tool_use_id) !== 'drop_call')
      .map((tool) => {
        if (actions.get(tool.tool_use_id) !== 'drop_result') return tool;
        const text = truncatedResultText(
          tool.text ?? '',
          tool.isError ?? false,
          headChars,
        );
        if ((tool.text ?? '') === text) return tool;
        const copy: ToolUse = {
          tool_use_id: tool.tool_use_id,
          tool: tool.tool,
          input: tool.input,
          text,
        };
        if (tool.isError) copy.isError = true;
        return copy;
      });
    const toolResults = (message.toolResults ?? [])
      .filter((result) => actions.get(result.tool_use_id) !== 'drop_call')
      .map((result) => {
        if (actions.get(result.tool_use_id) !== 'drop_result') return result;
        const text = truncatedResultText(result.text, result.isError ?? false, headChars);
        return text === result.text
          ? result
          : {
              tool_use_id: result.tool_use_id,
              text,
              isError: result.isError,
            };
      });
    if (
      !message.toolUses.some(
        (tool) => actions.get(tool.tool_use_id) === 'drop_call',
      ) &&
      !(message.toolResults ?? []).some(
        (result) => actions.get(result.tool_use_id) === 'drop_call',
      ) &&
      toolUses.every((tool, index) => tool === message.toolUses[index]) &&
      toolResults.every(
        (result, index) => result === message.toolResults?.[index],
      )
    ) {
      kept.push(message);
      continue;
    }
    if (message.text.trim().length === 0 && toolUses.length === 0 && toolResults.length === 0) {
      continue;
    }
    const rebuilt: Message = { role: message.role, text: message.text, toolUses };
    if (toolResults.length > 0) rebuilt.toolResults = toolResults;
    kept.push(rebuilt);
  }
  return kept;
}

/** Characters of text, tool input and tool output a message holds. */
export function messageChars(message: Message): number {
  let total = message.text.length;
  for (const tool of message.toolUses) {
    try {
      total += JSON.stringify(tool.input).length;
    } catch {
      total += 20;
    }
  }
  for (const result of message.toolResults ?? []) total += result.text.length;
  return total;
}

export function reductionRatio(result: Pick<CompactResult, 'stats'>): number {
  const { charsBefore, charsAfter } = result.stats;
  return charsBefore === 0 ? 0 : (charsBefore - charsAfter) / charsBefore;
}

function count(decisions: readonly CallDecision[], reason: CallDecision['reason']): number {
  return decisions.filter((decision) => decision.reason === reason).length;
}

/**
 * Compacts a transcript by asking Jev, for every tool call outside the pinned
 * first and newest messages, whether the call and whether its result must
 * stay. Calls beyond `maxCandidateCalls` share one vote per tool group
 * instead of being asked one by one, and old messages collapse accordingly.
 * The whole history (results omitted, fitted into `maxStateTokens`) is
 * sent as state with every batch of questions. Throws when Jev fails or the
 * history cannot be fitted; the caller decides whether to fall back.
 */
export async function compact(
  messages: readonly Message[],
  asker: JevAsker,
  options: CompactOptions = {},
): Promise<CompactResult> {
  const started = Date.now();
  const resolved = resolveOptions(options);
  const calls = collectToolCalls(messages, resolved.preserveRecentMessages);
  const { fine: candidates, grouped } = selectCandidates(calls, resolved.maxCandidateCalls);
  const groups = buildGroups(grouped);
  const charsBefore = messages.reduce((sum, message) => sum + messageChars(message), 0);

  let fitted: { tokens: number; stage: string } = { tokens: 0, stage: '' };
  let requestCount = 0;
  const answers = new Map<string, CallAnswer>();
  // [DEBUG temporaneo fork] fasi fit
  const debugLog = (payload: Record<string, unknown>): void => {
    void (async (): Promise<void> => {
      try {
        const line =
          JSON.stringify({ ts: new Date().toISOString(), ...payload }) + '\n';
        const fs = (await import('node:fs')) as typeof import('node:fs');
        fs.appendFileSync('C:/Users/Lucio/.claude/fast-jev-debug.log', line);
      } catch {
        console.error('[fast-jev-debug] ' + JSON.stringify(payload));
      }
    })();
  };
  if (candidates.length > 0 || groups.length > 0) {
    debugLog({
      phase: 'fit-start',
      messages: messages.length,
      calls: calls.length,
      candidates: candidates.length,
      groups: groups.length,
    });
    const state = fitState(messages, calls, resolved);
    debugLog({ phase: 'fit-done', stage: state.stage, tokens: state.tokens });
    fitted = state;
    // I gruppi vengono chiesti per primi: una richiesta propria, poi le call individuali.
    const groupBatches =
      groups.length > 0 ? batchCalls(groups, state.tokens, resolved, groupQuestions) : [];
    const callBatches = candidates.length > 0 ? batchCalls(candidates, state.tokens, resolved) : [];
    requestCount = groupBatches.length + callBatches.length;
    debugLog({ phase: 'requests-start', batches: requestCount });
    const answered = await Promise.all([
      ...groupBatches.map((batch) => askBatch(asker, state.state, batch, groupQuestions)),
      ...callBatches.map((batch) => askBatch(asker, state.state, batch, questionsFor)),
    ]);
    debugLog({ phase: 'requests-done' });
    for (const map of answered) for (const [id, answer] of map) answers.set(id, answer);
  } else {
    debugLog({ phase: 'no-candidates' });
  }

  // Grouped calls inherit their group's vote; individually-asked calls use
  // their own answers, and anything unanswered (pinned, or no request ran)
  // stays by default.
  const groupIdOfCall = new Map<string, string>();
  const groupIdByTool = new Map(groups.map((group) => [group.tool, group.id]));
  for (const call of grouped) {
    const groupId = groupIdByTool.get(call.tool);
    if (groupId) groupIdOfCall.set(call.id, groupId);
  }
  const decisions = calls.map((call) => {
    const groupId = groupIdOfCall.get(call.id);
    const answer = (groupId ? answers.get(groupId) : undefined) ?? answers.get(call.id);
    return decideCall(call, answer ?? { keepCall: 1, keepResult: 1 }, resolved);
  });
  const kept = applyDecisions(
    messages,
    decisions,
    calls,
    resolved.truncateHeadChars,
  );
  return {
    messages: kept,
    decisions,
    stats: {
      messagesBefore: messages.length,
      messagesAfter: kept.length,
      charsBefore,
      charsAfter: kept.reduce((sum, message) => sum + messageChars(message), 0),
      calls: calls.length,
      groupedCalls: grouped.length,
      kept: count(decisions, 'kept'),
      resultsDropped: count(decisions, 'result_dropped'),
      callsDropped: count(decisions, 'call_dropped'),
      pinned: count(decisions, 'pinned'),
      stateTokens: fitted.tokens,
      stateStage: fitted.stage,
      requests: requestCount,
      ms: Date.now() - started,
    },
  };
}
