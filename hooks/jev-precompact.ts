/**
 * [Local patch] Hook PreCompact ufficiale Claude Code: diagnostica Jev su transcript reale.
 *
 * Input  (stdin): JSON PreCompact { session_id, transcript_path, trigger, ... }
 * Output: report su ~/.claude/fast-jev-precompact.log + stderr (sempre exit 0, advisory).
 *
 * Non sostituisce l'output della compaction built-in (API non esposta): misura,
 * logga le decisioni Jev e verifica end-to-end il canale verso laya-serve.
 */
import fs from 'node:fs';
import { compact, resolveOptions } from '../src/compact.js';
import { DEFAULT_MODEL } from '../src/request.js';
import type { Message } from '../src/types.js';

const LOG_PATH = 'C:/Users/Lucio/.claude/fast-jev-precompact.log';
const BASE_URL = process.env.JEV_BASE_URL || 'http://127.0.0.1:8000/v1/systemone';
const MODEL = process.env.JEV_MODEL || DEFAULT_MODEL;
const MAX_STATE_TOKENS = 900;
const MAX_STATE_CHARS = 40_000;
const OVERALL_TIMEOUT_MS = 50_000;

type ToolUse = { tool_use_id: string; tool: string; input: unknown; text?: string; isError?: boolean };
type ToolResult = { tool_use_id: string; text: string; isError?: boolean };
type SessionMessage = { role: 'user' | 'assistant'; text: string; toolUses: ToolUse[]; toolResults?: ToolResult[] };

const logLine = (payload: Record<string, unknown>): void => {
  try {
    fs.appendFileSync(LOG_PATH, JSON.stringify({ ts: new Date().toISOString(), ...payload }) + '\n');
  } catch {
    /* il log non deve mai bloccare la compaction */
  }
};

const textOf = (content: unknown): string => {
  if (typeof content === 'string') return content;
  if (Array.isArray(content)) {
    return content
      .map((block) => (block && typeof block === 'object' && (block as { text?: string }).text) || '')
      .filter(Boolean)
      .join('\n');
  }
  return '';
};

/** Mappa una riga del transcript Claude Code su SessionMessage; null se non mappabile. */
function toSessionMessage(line: Record<string, unknown>): SessionMessage | null {
  const message = line.message as Record<string, unknown> | undefined;
  if (!message || typeof message !== 'object') return null;
  const role = message.role === 'assistant' ? 'assistant' : 'user';
  const content = message.content;
  if (typeof content === 'string') {
    return { role, text: content, toolUses: [] };
  }
  if (!Array.isArray(content)) return null;

  const out: SessionMessage = { role, text: '', toolUses: [], toolResults: [] };
  for (const block of content) {
    if (!block || typeof block !== 'object') continue;
    const b = block as Record<string, unknown>;
    if (b.type === 'text' && typeof b.text === 'string') {
      out.text += (out.text ? '\n' : '') + b.text;
    } else if (b.type === 'tool_use' && typeof b.id === 'string' && typeof b.name === 'string') {
      out.toolUses.push({ tool_use_id: b.id, tool: b.name, input: b.input ?? {} });
    } else if (b.type === 'tool_result' && typeof b.tool_use_id === 'string') {
      out.toolResults!.push({
        tool_use_id: b.tool_use_id,
        text: textOf(b.content),
        isError: b.is_error === true,
      });
    }
  }
  if (!out.text && out.toolUses.length === 0 && out.toolResults!.length === 0) return null;
  if (out.toolResults!.length === 0) delete out.toolResults;
  return out;
}

function readTranscript(path: string): SessionMessage[] {
  const raw = fs.readFileSync(path, 'utf8');
  const messages: SessionMessage[] = [];
  for (const line of raw.split('\n')) {
    if (!line.trim()) continue;
    try {
      const mapped = toSessionMessage(JSON.parse(line));
      if (mapped) messages.push(mapped);
    } catch {
      /* riga malformata: skip */
    }
  }
  return messages;
}

async function main(): Promise<void> {
  const stdin = fs.readFileSync(0, 'utf8');
  let payload: Record<string, unknown> = {};
  try {
    payload = JSON.parse(stdin);
  } catch {
    /* payload assente/malformato: prosegui senza trigger */
  }
  const transcriptPath = payload.transcript_path as string | undefined;
  const trigger = (payload.trigger as string) || 'unknown';
  if (!transcriptPath) {
    logLine({ phase: 'precompact-skip', reason: 'nessun transcript_path' });
    console.error('[fast-jev] PreCompact senza transcript_path: skip');
    process.exit(0);
  }

  const started = Date.now();
  const messages = readTranscript(transcriptPath);
  const resolved = resolveOptions({
    maxStateTokens: MAX_STATE_TOKENS,
    maxStateChars: MAX_STATE_CHARS,
  });

  const result = await Promise.race([
    compact(messages, {
      ask: async (state, questions) => {
        const body = JSON.stringify({ model: MODEL, state, questions });
        const response = await fetch(BASE_URL, {
          method: 'POST',
          headers: { 'content-type': 'application/json' },
          body,
        });
        const text = await response.text();
        if (!response.ok) throw new Error(`HTTP ${response.status}: ${text.slice(0, 200)}`);
        return JSON.parse(text) as { answers: Record<string, { noul?: number }> };
      },
    }, resolved),
    new Promise<never>((_, reject) =>
      setTimeout(() => reject(new Error('timeout compaction Jev')), OVERALL_TIMEOUT_MS),
    ),
  ]);

  const stats = result.stats as { requests?: number; stateTokens?: number; stateStage?: string };
  const durationMs = Date.now() - started;
  logLine({
    phase: 'precompact',
    trigger,
    messages: messages.length,
    requests: stats.requests ?? 0,
    stateTokens: stats.stateTokens ?? 0,
    stateStage: stats.stateStage ?? '',
    durationMs,
    decisions: result.decisions.slice(0, 50),
  });
  console.error(
    `[fast-jev] Jev OK: ${messages.length} messaggi, ${stats.requests ?? 0} richieste, ` +
      `state ${stats.stateTokens ?? 0} token/${stats.stateStage ?? '?'} in ${durationMs}ms ` +
      `(trigger: ${trigger})`,
  );
  process.exit(0);
}

main().catch((error: unknown) => {
  const message = error instanceof Error ? error.message : String(error);
  logLine({ phase: 'precompact-error', error: message });
  console.error(`[fast-jev] ${message}`);
  process.exit(0); // advisory: non bloccare MAI la compaction
});
