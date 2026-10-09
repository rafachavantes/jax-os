import { NextResponse } from "next/server";
import { LIMITS } from "../../../../lib/workflow";
import { readJsonCapped, requireSameOrigin } from "../../../../server/api";
import { parseAnswerBody } from "../../../../server/collectors/workflow-events";
import { injectAnswers } from "../../../../server/collectors/workflow-tmux";
import { codexQueueMessage } from "../../../../server/collectors/workflow-codex";
import { isAnswerWatcherAlive, writeAnswerLine } from "../../../../server/collectors/workflow-callbacks";
import { readGeneralSettings } from "../../../../server/settings";
import { getDb } from "../../../../server/db";
import { claimAnswer, claimFreeformAnswer, failAnswerFinalization, finishAnswer } from "../../../../server/db/workflows";

export function loadEventType(db: ReturnType<typeof getDb>, id: number): string | null {
  const row = db.prepare("SELECT type FROM workflow_events WHERE id = ?").get(id) as { type: string } | undefined;
  return row?.type ?? null;
}

export type AnswerRouteDeps = {
  getDb: typeof getDb;
  readGeneralSettings: typeof readGeneralSettings;
  loadEventType: typeof loadEventType;
  claimAnswer: typeof claimAnswer;
  claimFreeformAnswer: typeof claimFreeformAnswer;
  injectAnswers: typeof injectAnswers;
  codexQueueMessage: typeof codexQueueMessage;
  isAnswerWatcherAlive: typeof isAnswerWatcherAlive;
  writeAnswerLine: typeof writeAnswerLine;
  finishAnswer: typeof finishAnswer;
  failAnswerFinalization: typeof failAnswerFinalization;
};

const systemDeps: AnswerRouteDeps = {
  getDb, readGeneralSettings, loadEventType, claimAnswer, claimFreeformAnswer, injectAnswers, codexQueueMessage, isAnswerWatcherAlive, writeAnswerLine, finishAnswer, failAnswerFinalization,
};

const INJECT_ERRORS = new Set([
  "pane target is not live",
  "tmux answer injection failed",
  "reply sent, confirmation key failed",
  "codex queue cli missing",
  "codex queue failed",
  "codex queue timeout",
  "invalid harness session",
  "answer watcher not live",
]);

export async function handleAnswerPost(req: Request, deps: AnswerRouteDeps = systemDeps) {
  const guard = requireSameOrigin(req);
  if (guard) return guard;
  const read = await readJsonCapped(req, LIMITS.requestBytes);
  if (!read.ok) return NextResponse.json({ ok: false, error: read.error });
  const parsed = parseAnswerBody(read.value);
  if (!parsed.ok) return NextResponse.json({ ok: false, error: `invalid answer: ${parsed.error}` });

  let db: ReturnType<typeof getDb>;
  let eventType: string | null;
  try {
    db = deps.getDb();
    eventType = deps.loadEventType(db, parsed.answer.question_event_id);
  } catch {
    return NextResponse.json({ ok: false, error: "answer claim failed" });
  }

  if (eventType !== "question" && eventType !== "turn-stopped") {
    return NextResponse.json({ ok: false, error: "not-answerable" });
  }

  let claimed: ReturnType<typeof claimAnswer> | ReturnType<typeof claimFreeformAnswer>;
  try {
    claimed = eventType === "question" ? deps.claimAnswer(db, parsed.answer) : deps.claimFreeformAnswer(db, parsed.answer);
  } catch {
    return NextResponse.json({ ok: false, error: "answer claim failed" });
  }
  if (!claimed.ok) return NextResponse.json({ ok: false, error: claimed.reason });

  const { claim } = claimed;
  // D5 — every delivered freeform reply is attributed, so the tech lead reading it knows the words
  // after the prefix are the owner's own. Decision 11: read live, per request; a missing/unreadable
  // settings file or an empty ownerName drops the name, never the prefix.
  const settings = deps.readGeneralSettings();
  const ownerName = settings.ok ? settings.data.ownerName : "";
  const rafaPrefix = ownerName ? `[Jax OS · ${ownerName}] ` : "[Jax OS] ";
  let ok = false;
  let error: string | null = null;
  try {
    if (claim.kind === "structured") {
      await deps.injectAnswers(claim.pane ?? "", claim.question_shapes ?? [], claim.answers ?? [], claim.tmux_incarnation ?? "");
    } else if (claim.emitter === "codex-stop") {
      await deps.codexQueueMessage(claim.harnessSession ?? "", `${rafaPrefix}${claim.text ?? ""}`);
    } else {
      // Review F1 (plan round 1): an expired/absent watcher must fail the click, not accept a file nobody reads.
      if (!deps.isAnswerWatcherAlive(claim.harnessSession ?? "", claim.question_event_id)) {
        throw new Error("answer watcher not live");
      }
      deps.writeAnswerLine(claim.harnessSession ?? "", claim.question_event_id, `${rafaPrefix}${claim.text ?? ""}`);
    }
    ok = true;
  } catch (e) {
    error = e instanceof Error && INJECT_ERRORS.has(e.message) ? e.message : "tmux answer injection failed";
  }

  try {
    deps.finishAnswer(db, claim, ok, error);
  } catch {
    try {
      deps.failAnswerFinalization(db, claim);
    } catch {
      // Startup recovery owns the still-injecting row if the DB remains unavailable.
    }
    return NextResponse.json({ ok: false, error: "answer finalization failed" });
  }
  if (!ok) return NextResponse.json({ ok: false, error });
  return NextResponse.json({
    ok: true,
    data: {
      question_event_id: claim.question_event_id,
      tool_use_id: claim.tool_use_id,
      mutation_id: claim.mutation_id,
      kind: claim.kind,
    },
  });
}
