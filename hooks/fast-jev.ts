import type {
  On,
  PluginOptions,
  Register,
  SessionMessage,
  ToolResultSummary,
  ToolUseSummary,
  TurnCompleteInput,
} from 'claude-code';

import { compact, reductionRatio, resolveOptions } from '../src/compact.js';
import { buildJevRequest, DEFAULT_MODEL, parseJevResponse } from '../src/request.js';
import type {
  CompactOptions,
  CompactResult,
  JevAsker,
  Message,
  ToolResult,
  ToolUse,
} from '../src/types.js';

const HOOK_DEFAULTS = {
  compactAtPercent: 60,
  minReductionRatio: 0.25,
  model: DEFAULT_MODEL,
};

export type HookFetchInit = {
  method?: string;
  headers?: Record<string, string>;
  body?: string;
};

export type HookFetchResponse = {
  status: number;
  ok: boolean;
  text: string;
};

/** The shape of `$.http.fetch`, so the hook can be driven without an engine. */
export type HookFetch = (url: string, init?: HookFetchInit) => Promise<HookFetchResponse>;

export type HookConfig = CompactOptions & {
  apiKey?: string;
  compactAtPercent: number;
  minReductionRatio: number;
  model: string;
  /** Base URL of the Jev server (local-patch: default http://127.0.0.1:8787). */
  baseUrl?: string;
};

function optionNumber(options: PluginOptions, key: string, fallback: number): number {
  const value = options[key];
  return typeof value === 'number' && Number.isFinite(value) ? value : fallback;
}

function optionString(options: PluginOptions, key: string): string | undefined {
  const value = options[key];
  return typeof value === 'string' && value.length > 0 ? value : undefined;
}

/** Reads the plugin's `userConfig` values; anything missing takes the defaults. */
export function resolveHookConfig(options: PluginOptions): HookConfig {
  const numbers: Partial<Omit<CompactOptions, 'goal'>> = {};
  for (const key of [
    'keepThreshold',
    'preserveRecentMessages',
    'maxStateTokens',
    'maxStateChars',
    'maxRequestTokens',
    'truncateHeadChars',
  ] as const) {
    const value = options[key];
    if (typeof value === 'number' && Number.isFinite(value)) numbers[key] = value;
  }
  const config: HookConfig = {
    ...numbers,
    compactAtPercent: optionNumber(options, 'compactAtPercent', HOOK_DEFAULTS.compactAtPercent),
    minReductionRatio: optionNumber(
      options,
      'minReductionRatio',
      HOOK_DEFAULTS.minReductionRatio,
    ),
    model: optionString(options, 'model') ?? HOOK_DEFAULTS.model,
  };
  const apiKey = optionString(options, 'apiKey');
  if (apiKey) config.apiKey = apiKey;
  const goal = optionString(options, 'goal');
  if (goal) config.goal = goal;
  const baseUrl = optionString(options, 'baseUrl');
  if (baseUrl) config.baseUrl = baseUrl;
  // Laya: contesto piccolo → budget stato ridotto se non impostato esplicitamente
  if (config.baseUrl && numbers.maxStateTokens === undefined) {
    config.maxStateTokens = LAYA_MAX_STATE_TOKENS;
  }
  if (config.baseUrl && numbers.maxStateChars === undefined) {
    config.maxStateChars = LAYA_MAX_STATE_CHARS;
  }
  return config;
}

/** Local patch: Jev-protocol endpoint defaults to the user's local Laya server. */
const DEFAULT_BASE_URL = 'http://127.0.0.1:8000/v1/systemone';

/** Local patch: Laya has a small context — reduced state budget when not set explicitly. */
const LAYA_MAX_STATE_TOKENS = 900;

/** Local patch: hard character cap for the serialized state. Token estimates undercount
 *  symbol/URL/JSON-heavy content, and laya-serve answers 413 past 50 000 state chars
 *  (serve.py MAX_STATE_CHARS). 40 000 leaves headroom for JSON escaping. */
const LAYA_MAX_STATE_CHARS = 40_000;

async function getBaseUrl(
  $: {
    env: { get: (name: string) => Promise<string | undefined> };
    settings: { read: () => Promise<Readonly<Record<string, unknown>>> };
  },
  config: HookConfig,
): Promise<string> {
  if (config.baseUrl) return config.baseUrl;
  const fromEnv = await $.env.get('JEV_BASE_URL');
  if (fromEnv) return fromEnv;
  const settings = await $.settings.read();
  const env = settings['env'];
  if (env && typeof env === 'object') {
    const value = (env as Record<string, unknown>)['JEV_BASE_URL'];
    if (typeof value === 'string' && value) return value;
  }
  return DEFAULT_BASE_URL;
}

/** A `JevAsker` over the engine's `$.http.fetch` — Jev protocol (hosted TypeSafe or local Laya). */
export function jevAsker(
  fetchFn: HookFetch,
  apiKey: string,
  model: string,
  baseUrl?: string,
): JevAsker {
  return {
    async ask(state, questions) {
      const request = buildJevRequest({ apiKey, model, baseUrl }, state, questions);
      // [DEBUG temporaneo fork] log diagnostico su ogni richiesta — rimuovere dopo analisi
      try {
        const debugLine = JSON.stringify({
          ts: new Date().toISOString(),
          url: request.url,
          bodyLen: request.body.length,
          stateLen: typeof state === 'string' ? state.length : JSON.stringify(state).length,
          questions: Object.keys(questions ?? {}),
        });
        try {
          const fs = await import('node:fs');
          fs.appendFileSync('C:/Users/Lucio/.claude/fast-jev-debug.log', debugLine + '\n');
        } catch {
          console.error('[fast-jev-debug] ' + debugLine);
        }
      } catch {}
      const response = await fetchFn(request.url, {
        method: request.method,
        headers: request.headers,
        body: request.body,
      });
      // [DEBUG temporaneo fork] risposta (status + inizio body)
      try {
        console.error(
          `[fast-jev-debug] status=${response.status} resp=${response.text.slice(0, 200)}`,
        );
      } catch {}
      return parseJevResponse(response.status, response.ok, response.text);
    },
  };
}

function toolUseSummary(tool: ToolUse): ToolUseSummary {
  const summary: ToolUseSummary = {
    tool_use_id: tool.tool_use_id,
    tool: tool.tool,
    input: tool.input,
  };
  if (tool.text !== undefined) summary.text = tool.text;
  if (tool.isError) summary.isError = true;
  return summary;
}

function toolResultSummary(result: ToolResult): ToolResultSummary {
  return {
    tool_use_id: result.tool_use_id,
    text: result.text,
    isError: result.isError ?? false,
  };
}

/**
 * Maps the library's output back onto session messages. Whatever came back
 * unchanged (a message, a tool use, a tool result) is the engine's own object,
 * handle included; anything rebuilt is a fresh message without a handle, so the
 * engine takes the edited content instead of its original.
 */
export function toSessionMessages(
  input: readonly SessionMessage[],
  output: readonly Message[],
): SessionMessage[] {
  const messages = new Map<Message, SessionMessage>();
  const uses = new Map<ToolUse, ToolUseSummary>();
  const results = new Map<ToolResult, ToolResultSummary>();
  for (const message of input) {
    messages.set(message, message);
    for (const tool of message.toolUses) uses.set(tool, tool);
    for (const result of message.toolResults ?? []) results.set(result, result);
  }
  return output.map((message) => {
    const own = messages.get(message);
    if (own) return own;
    const rebuilt: SessionMessage = {
      role: message.role,
      text: message.text,
      toolUses: message.toolUses.map((tool) => uses.get(tool) ?? toolUseSummary(tool)),
    };
    if (message.toolResults && message.toolResults.length > 0) {
      rebuilt.toolResults = message.toolResults.map(
        (result) => results.get(result) ?? toolResultSummary(result),
      );
    }
    return rebuilt;
  });
}

export type SessionCompaction = {
  result: CompactResult;
  messages: SessionMessage[];
};

/** Runs the library over a session transcript; throws when the key is missing or Jev fails. */
export async function compactSession(
  messages: readonly SessionMessage[],
  config: HookConfig,
  fetchFn: HookFetch,
): Promise<SessionCompaction> {
  if (config.baseUrl) {
    // Server locale (Laya): nessuna API key richiesta, stesso protocollo Jev su /v1/systemone
    const result = await compact(
      messages,
      jevAsker(fetchFn, config.apiKey ?? '', config.model, config.baseUrl),
      config,
    );
    return { result, messages: toSessionMessages(messages, result.messages) };
  }
  if (!config.apiKey) throw new Error('TYPESAFE_API_KEY is not configured');
  const result = await compact(messages, jevAsker(fetchFn, config.apiKey, config.model), config);
  return { result, messages: toSessionMessages(messages, result.messages) };
}

function percent(ratio: number): string {
  return `${Math.round(ratio * 100)}%`;
}

export function summarize(result: CompactResult): string {
  const { stats } = result;
  const parts = [
    stats.kept > 0 ? `${stats.kept} kept` : '',
    stats.resultsDropped > 0 ? `${stats.resultsDropped} results truncated` : '',
    stats.callsDropped > 0 ? `${stats.callsDropped} call_dropped` : '',
    stats.pinned > 0 ? `${stats.pinned} pinned` : '',
  ].filter(Boolean);
  return `${percent(reductionRatio(result))} reduction; ${
    parts.join(', ') || 'no tool calls'
  }; state ~${stats.stateTokens} tokens (${stats.stateStage}) in ${stats.requests} request(s)`;
}

const UI_LOG_MAX_CHARS = 4096;

export function decisionLog(result: CompactResult): string {
  return result.decisions
    .filter((d) => d.reason !== 'pinned')
    .map(
      (d) =>
        `${d.id}:${d.tool}:${d.action}/call=${d.keepCall.toFixed(2)}/result=${d.keepResult.toFixed(2)}`,
    )
    .join(' ');
}

export function decisionLogLines(
  result: CompactResult,
  maxChars: number = UI_LOG_MAX_CHARS,
): string[] {
  const entries = decisionLog(result).split(' ').filter(Boolean);
  if (entries.length === 0) return ['decisions: (none)'];
  const chunks: string[] = [];
  let current = '';
  for (const entry of entries) {
    const next = current ? `${current} ${entry}` : entry;
    if (current && next.length > maxChars - 24) {
      chunks.push(current);
      current = entry;
    } else current = next;
  }
  chunks.push(current);
  return chunks.map((chunk, index) =>
    chunks.length === 1
      ? `decisions: ${chunk}`
      : `decisions (${index + 1}/${chunks.length}): ${chunk}`,
  );
}

async function getApiKey(
  $: {
    env: { get: (name: string) => Promise<string | undefined> };
    settings: { read: () => Promise<Readonly<Record<string, unknown>>> };
  },
  config: HookConfig,
): Promise<string | undefined> {
  if (config.apiKey) return config.apiKey;
  const fromEnv = await $.env.get('TYPESAFE_API_KEY');
  if (fromEnv) return fromEnv;
  const settings = await $.settings.read();
  const env = settings['env'];
  if (env && typeof env === 'object') {
    const value = (env as Record<string, unknown>)['TYPESAFE_API_KEY'];
    if (typeof value === 'string' && value) return value;
  }
  return undefined;
}

function notify(
  $: {
    ui: {
      log: (text: string) => void;
      toast: (text: string, options?: { timeoutMs?: number }) => void;
    };
  },
  text: string,
): void {
  $.ui.log(text);
  $.ui.toast(text, { timeoutMs: 15_000 });
}

export const register: Register = (on: On, options: PluginOptions) => {
  // [DEBUG temporaneo fork] register chiamato dal loader = modulo caricato
  try {
    const line =
      JSON.stringify({ ts: new Date().toISOString(), phase: 'register' }) + '\n';
    void (async () => {
      try {
        const fs = (await import('node:fs')) as typeof import('node:fs');
        fs.appendFileSync('C:/Users/Lucio/.claude/fast-jev-debug.log', line);
      } catch {
        console.error('[fast-jev-debug] ' + line);
      }
    })();
  } catch {}
  const configured = resolveHookConfig(options);
  let compacting = false;

  on('session.compact', async ($, event, next) => {
    try {
      // [DEBUG temporaneo fork] ingresso hook
      try {
        const line =
          JSON.stringify({
            ts: new Date().toISOString(),
            phase: 'hook-start',
            messages: event.messages.length,
          }) + '\n';
        try {
          const fs = await import('node:fs');
          fs.appendFileSync('C:/Users/Lucio/.claude/fast-jev-debug.log', line);
        } catch {
          console.error('[fast-jev-debug] ' + line);
        }
      } catch {}
      const config = {
        ...configured,
        apiKey: await getApiKey($, configured),
        baseUrl: await getBaseUrl($, configured),
      };
      const { result, messages } = await compactSession(event.messages, config, async (url, init) => {
        const response = await $.http.fetch(url, init);
        return { status: response.status, ok: response.ok, text: response.text };
      });
      for (const line of decisionLogLines(result)) $.ui.log(line);
      if (reductionRatio(result) < config.minReductionRatio) {
        notify(
          $,
          `fallback to built-in summary (below ${percent(config.minReductionRatio)} minimum: ${summarize(result)})`,
        );
        return next(event);
      }
      notify(
        $,
        `kept ${messages.length}/${event.messages.length} messages, no summary (${summarize(result)})`,
      );
      return { messages };
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      // [DEBUG temporaneo fork] errore reale della compaction su file
      try {
        const line =
          JSON.stringify({ ts: new Date().toISOString(), phase: 'compact', error: message }) + '\n';
        try {
          const fs = await import('node:fs');
          fs.appendFileSync('C:/Users/Lucio/.claude/fast-jev-debug.log', line);
        } catch {
          console.error('[fast-jev-debug] ' + line);
        }
      } catch {}
      notify($, `fallback to built-in summary (${message})`);
      return next(event);
    }
  });

  on('turn.complete', async ($, event: TurnCompleteInput, next) => {
    if (compacting) return next(event);
    try {
      const { context } = await $.session.usage();
      // [DEBUG temporaneo fork] soglia e percentuale contesto a ogni fine turno
      try {
        const line =
          JSON.stringify({
            ts: new Date().toISOString(),
            phase: 'turn-complete',
            percent: context.percent ?? 0,
            threshold: configured.compactAtPercent,
          }) + '\n';
        try {
          const fs = await import('node:fs');
          fs.appendFileSync('C:/Users/Lucio/.claude/fast-jev-debug.log', line);
        } catch {
          console.error('[fast-jev-debug] ' + line);
        }
      } catch {}
      if ((context.percent ?? 0) < configured.compactAtPercent) return next(event);
      compacting = true;
      await $.session.compact();
    } catch (error) {
      $.ui.log(
        `auto-compact skipped (${error instanceof Error ? error.message : String(error)})`,
      );
    } finally {
      compacting = false;
    }
    return next(event);
  });
};

export { resolveOptions };
