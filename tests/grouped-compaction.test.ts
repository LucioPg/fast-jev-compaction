import { describe, expect, it } from 'vitest';
import {
  batchCalls,
  collectToolCalls,
  compact,
  fitState,
  resolveOptions,
  selectCandidates,
  type JevAsker,
  type JevQuestions,
  type Message,
} from '../src/index.js';

// Transcript sintetico con N call alternate Bash/Edit, risultati da ~2000 char.
function bigTranscript(calls: number): Message[] {
  const messages: Message[] = [{ role: 'user', text: 'refactor the parser', toolUses: [] }];
  for (let i = 0; i < calls; i++) {
    messages.push({
      role: 'assistant',
      text: `step ${i}`,
      toolUses: [{ tool_use_id: `u${i}`, tool: i % 2 === 0 ? 'Bash' : 'Edit', input: { i } }],
    });
    messages.push({
      role: 'user',
      text: '',
      toolUses: [],
      toolResults: [{ tool_use_id: `u${i}`, text: `output ${i}: ${'x'.repeat(2000)}` }],
    });
  }
  return messages;
}

type Seen = { state: unknown; questions: string[] };

function fakeJev(answer: (name: string) => number, seen: Seen[]): JevAsker {
  return {
    async ask(state, questions: JevQuestions) {
      seen.push({ state, questions: Object.keys(questions) });
      return {
        answers: Object.fromEntries(
          Object.keys(questions).map((key) => [key, { type: 'noul' as const, noul: answer(key) }]),
        ),
      };
    },
  };
}

describe('candidate selection (macro-categorie)', () => {
  it('separa le call non-pinned più recenti da quelle vecchie', () => {
    const messages = bigTranscript(40);
    const calls = collectToolCalls(messages, 2);
    const { fine, grouped } = selectCandidates(calls, 8);
    expect(fine.length).toBe(8);
    expect(grouped.length).toBe(calls.filter((c) => !c.pinned).length - 8);
    // budget alto: nessuna call raggruppata (comportamento attuale)
    expect(selectCandidates(calls, 1000).grouped.length).toBe(0);
  });
});

describe('fitState con call raggruppate', () => {
  it('200 call stanno nel budget Laya grazie a una nota aggregata per tool', () => {
    const messages = bigTranscript(200);
    const calls = collectToolCalls(messages, 2);
    const fitted = fitState(
      messages,
      calls,
      resolveOptions({ maxStateTokens: 3000, maxStateChars: 40_000, preserveRecentMessages: 2 }),
    );
    expect(fitted.tokens).toBeLessThanOrEqual(3000);
    const note = fitted.state.history.find((entry) =>
      entry.text.includes('earlier Bash tool calls'),
    );
    expect(note).toBeDefined();
    // le call raggruppate non compaiono più come one-liner (t1 = la più vecchia)
    const serialized = JSON.stringify(fitted.state);
    expect(serialized).not.toContain('"t1"');
  });

  it('con poche call nessuna viene raggruppata', () => {
    const messages = bigTranscript(10);
    const calls = collectToolCalls(messages, 2);
    const fitted = fitState(messages, calls, resolveOptions({ preserveRecentMessages: 2 }));
    expect(JSON.stringify(fitted.state)).not.toContain('earlier Bash tool calls');
  });
});

describe('compact con decisioni a gruppi', () => {
  it('un voto per gruppo sulle call vecchie, un voto per call sulle recenti', async () => {
    const messages = bigTranscript(60);
    const seen: Seen[] = [];
    const result = await compact(
      messages,
      fakeJev((name) => (name.startsWith('result_') ? 0 : 0.9), seen),
      { maxStateTokens: 3000, preserveRecentMessages: 2, maxCandidateCalls: 8 },
    );
    const asked = seen.flatMap((s) => s.questions);
    expect(asked.filter((q) => /^call_g\d+$/.test(q)).length).toBe(2); // Bash + Edit
    expect(asked.filter((q) => /^call_t\d+$/.test(q)).length).toBe(8);
    for (const s of seen) expect(s.questions.length).toBeLessThanOrEqual(64);
    expect(result.stats.groupedCalls).toBe(
      result.stats.calls - result.stats.pinned - 8,
    );
    expect(result.stats.requests).toBe(seen.length);
  });

  it('applica la decisione del gruppo a ogni call raggruppata', async () => {
    const messages = bigTranscript(40);
    const seen: Seen[] = [];
    const result = await compact(
      messages,
      fakeJev((name) => (name.startsWith('result_') ? 0 : 0.9), seen),
      { maxStateTokens: 3000, preserveRecentMessages: 2, maxCandidateCalls: 4 },
    );
    // keepResult 0 ovunque → tutti i risultati vecchi troncati
    const truncated = result.messages.filter((m) =>
      (m.toolResults ?? []).some((r) => r.text.includes('truncated')),
    );
    expect(truncated.length).toBeGreaterThan(0);
    // le call restano (keepCall 0.9)
    expect(result.stats.callsDropped).toBe(0);
  });

  it('300 call: mai più di 64 domande per richiesta, gruppi al primo batch', async () => {
    const messages = bigTranscript(300);
    const seen: Seen[] = [];
    await compact(
      messages,
      fakeJev(() => 0.9, seen),
      { maxStateTokens: 3000, preserveRecentMessages: 2 },
    );
    expect(seen.length).toBeGreaterThan(1);
    for (const s of seen) expect(s.questions.length).toBeLessThanOrEqual(64);
    expect(seen[0]!.questions.some((q) => /^call_g\d+$/.test(q))).toBe(true);
  });

  it('storia piccola: percorso per-call invariato', async () => {
    const messages = bigTranscript(10);
    const seen: Seen[] = [];
    const result = await compact(messages, fakeJev(() => 0.9, seen), {
      maxStateTokens: 3000,
      preserveRecentMessages: 2,
    });
    expect(seen.flatMap((s) => s.questions).some((q) => /^call_g\d+$/.test(q))).toBe(false);
    expect(result.stats.groupedCalls).toBe(0);
  });
});

describe('batchCalls cap per richiesta', () => {
  it('spezza oltre 32 item così una richiesta non supera 64 domande', () => {
    const messages = bigTranscript(80);
    const calls = collectToolCalls(messages, 2);
    const { fine } = selectCandidates(calls, 80);
    const resolved = resolveOptions({ maxRequestTokens: 30_000 });
    const batches = batchCalls(fine, 500, resolved);
    expect(batches.length).toBeGreaterThan(1);
    for (const batch of batches) expect(batch.length).toBeLessThanOrEqual(32);
  });
});
