"use client";

import { useQuery, useQueryClient } from "@tanstack/react-query";
import { fetchEnvelope } from "./api";
import {
  applySavedOrder, dropEntry, moveEntry, setEntryWidth, visibleLayout,
  type CockpitSnapshot, type LayoutEntry, type PrScope, type ProjectWidth,
} from "./cockpitLayout";

export type CockpitWriteData = { conflict: boolean; snapshot: CockpitSnapshot };

export function useCockpitSnapshot() {
  return useQuery({
    queryKey: ["mission", "cockpit-prefs"],
    queryFn: ({ signal }) => fetchEnvelope<CockpitSnapshot>("/api/mission/cockpit-prefs", signal),
    refetchInterval: 10_000,
  });
}

async function postCockpit<T>(body: Record<string, unknown>): Promise<T> {
  const res = await fetch("/api/mission/cockpit-prefs", {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(body),
  });
  const json: unknown = await res.json().catch(() => null);
  if (!res.ok || !json || typeof json !== "object" || !("ok" in json)) throw new Error("unavailable");
  const env = json as { ok: boolean; data?: T; error?: string };
  if (!env.ok || env.data === undefined) throw new Error(env.error || "unavailable");
  return env.data;
}

export function saveLayout(expectedRevision: number, projects: LayoutEntry[]): Promise<CockpitWriteData> {
  return postCockpit<CockpitWriteData>({ action: "layout-save", expectedRevision, projects });
}

export function resetLayout(expectedRevision: number): Promise<CockpitWriteData> {
  return postCockpit<CockpitWriteData>({ action: "layout-reset", expectedRevision });
}

export function savePrScope(scope: PrScope): Promise<CockpitSnapshot> {
  return postCockpit<CockpitSnapshot>({ action: "pr-scope", scope });
}

export function useCockpitWrites() {
  const client = useQueryClient();
  const apply = (snapshot: CockpitSnapshot) => client.setQueryData(["mission", "cockpit-prefs"], { ok: true, data: snapshot });
  return { apply, saveLayout, resetLayout, savePrScope };
}

// ---- Organize draft (plan Task 2) ----------------------------------------------------------
// The draft is deliberately separate from the polled snapshot: a dirty draft survives every new
// server snapshot, and only an explicit success, Cancel or Reload replaces it.
export type OrganizeStatus = "view" | "dirty" | "saving" | "error" | "conflict";

export type OrganizeState = {
  draft: LayoutEntry[] | null;
  baseRevision: number | null;
  baseline: CockpitSnapshot | null;
  status: OrganizeStatus;
  error: string | null;
};

export const initialOrganizeState: OrganizeState = {
  draft: null, baseRevision: null, baseline: null, status: "view", error: null,
};

export function enterOrganize(snapshot: CockpitSnapshot, cards: { dir: string }[]): OrganizeState {
  const ordered = applySavedOrder(cards, snapshot.layout);
  return {
    draft: visibleLayout(ordered, snapshot.layout),
    baseRevision: snapshot.revision,
    baseline: snapshot,
    status: "dirty",
    error: null,
  };
}

// A poll NEVER overwrites a dirty draft; in view mode it becomes the latest clean baseline.
export function applySnapshot(state: OrganizeState, snapshot: CockpitSnapshot): OrganizeState {
  if (state.draft !== null) return state;
  return { ...state, baseline: snapshot };
}

export function cancelOrganize(snapshot: CockpitSnapshot): OrganizeState {
  return { ...initialOrganizeState, baseline: snapshot };
}

// While a layout write is in flight the visible draft is exactly what Save submitted: freezing
// edits here (not only in the UI) keeps a newer width/order from being replaced by the response.
const editable = (state: OrganizeState) => state.draft !== null && state.status !== "saving";

export function draftMove(state: OrganizeState, project: string, delta: -1 | 1): OrganizeState {
  if (!editable(state)) return state;
  return { ...state, draft: moveEntry(state.draft!, project, delta), status: "dirty", error: null };
}

export function draftDrop(state: OrganizeState, dragged: string, target: string): OrganizeState {
  if (!editable(state)) return state;
  return { ...state, draft: dropEntry(state.draft!, dragged, target), status: "dirty", error: null };
}

export function draftWidth(state: OrganizeState, project: string, width: ProjectWidth): OrganizeState {
  if (!editable(state)) return state;
  return { ...state, draft: setEntryWidth(state.draft!, project, width), status: "dirty", error: null };
}

export function beginSave(state: OrganizeState): OrganizeState {
  if (!editable(state)) return state;
  return { ...state, status: "saving", error: null };
}

export function saveSucceeded(snapshot: CockpitSnapshot): OrganizeState {
  return { ...initialOrganizeState, baseline: snapshot };
}

export function saveFailed(state: OrganizeState, error: string): OrganizeState {
  return { ...state, status: "error", error };
}

export function saveConflicted(state: OrganizeState, snapshot: CockpitSnapshot): OrganizeState {
  return { ...state, status: "conflict", baseline: snapshot, error: null };
}

// Reload explicitly replaces a conflicted draft with the latest server state (staying in
// Organize with a fresh revision); it never silently discards the draft.
export function reloadOrganize(snapshot: CockpitSnapshot, cards: { dir: string }[]): OrganizeState {
  return enterOrganize(snapshot, cards);
}
