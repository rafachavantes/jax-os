"use client";

import { createContext, useCallback, useContext, useState, type ReactNode } from "react";
import { answerEventId } from "@/lib/mission";

// Unfinished answers, held ABOVE the cards (spec §11). A board card can be recreated while the
// owner is typing — a second session opening or closing re-groups the project — and a card's own
// state would go with it. Drafts are keyed by the QUESTION's event id, so the same question gets
// its draft back wherever it renders next, and a different question never inherits one.
export type AnswerDraftStore = {
  get(key: string): unknown;
  set(key: string, value: unknown): void;
  // Drop every draft for one question event (after it was answered successfully).
  clearEvent(eventId: number): void;
};

export const draftKeys = {
  freeform: (eventId: number) => `freeform:${eventId}`,
  question: (eventId: number) => `question:${eventId}`,
};

export function createAnswerDraftStore(): AnswerDraftStore {
  const drafts = new Map<string, unknown>();
  return {
    get: (key) => drafts.get(key),
    set: (key, value) => { drafts.set(key, value); },
    clearEvent: (eventId) => {
      drafts.delete(draftKeys.freeform(eventId));
      drafts.delete(draftKeys.question(eventId));
    },
  };
}

// The pure write-through path every form uses (tested without React).
export function initialDraft<T>(store: AnswerDraftStore | null, key: string, initial: () => T): T {
  return (store?.get(key) as T | undefined) ?? initial();
}

export function draftUpdater<T>(store: AnswerDraftStore | null, key: string, setLocal: (v: T) => void): (next: T) => void {
  return (next) => {
    store?.set(key, next);
    setLocal(next);
  };
}

// A successful answer clears that event's draft; a failure keeps it.
export function settleAnswer(store: AnswerDraftStore | null, ok: boolean, body: unknown): void {
  if (!ok) return;
  const eventId = answerEventId(body);
  if (eventId !== null) store?.clearEvent(eventId);
}

const AnswerDraftContext = createContext<AnswerDraftStore | null>(null);

export function AnswerDraftsProvider({ children, store }: { children?: ReactNode; store?: AnswerDraftStore }) {
  const [own] = useState(createAnswerDraftStore);
  return <AnswerDraftContext.Provider value={store ?? own}>{children}</AnswerDraftContext.Provider>;
}

export function useAnswerDraftStore(): AnswerDraftStore | null {
  return useContext(AnswerDraftContext);
}

// A draft value for `key`: starts from the stored draft (else `initial`), and every update is
// written through to the store. Without a provider it is plain local state. Callers render the
// form with `key={eventId}` so a different question is a fresh instance.
export function useAnswerDraft<T>(key: string, initial: () => T): [T, (next: T) => void] {
  const store = useAnswerDraftStore();
  const [value, setValue] = useState<T>(() => initialDraft(store, key, initial));
  const update = useCallback(draftUpdater(store, key, setValue), [store, key]);
  return [value, update];
}
