import { QueryClient } from "@tanstack/react-query";
import { describe, expect, it } from "vitest";
import type { Envelope } from "@/lib/api";
import type { HubEnvelopeData, ProjectPrefs } from "@/server/db/workflows";
import { HUB_QUERY_KEY, nextPrefs, postPrefsPatch } from "./prefsAction";

function hubFixture(prefs: Record<string, ProjectPrefs>): Envelope<HubEnvelopeData> {
  return { ok: true, data: { byProject: {}, historicalTruncated: false, codexSource: "ok", tmuxSource: "ok", prefs, settings: { archiveAfterDays: 30 } } };
}

describe("nextPrefs (mirrors setProjectPrefs' own patch semantics, src/server/db/workflows.ts)", () => {
  it("hidden:true stamps hiddenAt, hidden:false clears it, pinned never touches hiddenAt", () => {
    const now = new Date("2026-09-19T12:00:00.000Z");
    expect(nextPrefs({ pinned: false, hiddenAt: null }, { hidden: true }, now)).toEqual({ pinned: false, hiddenAt: now.toISOString() });
    expect(nextPrefs({ pinned: false, hiddenAt: "2026-01-01T00:00:00.000Z" }, { hidden: false }, now)).toEqual({ pinned: false, hiddenAt: null });
    expect(nextPrefs({ pinned: false, hiddenAt: "2026-01-01T00:00:00.000Z" }, { pinned: true }, now)).toEqual({ pinned: true, hiddenAt: "2026-01-01T00:00:00.000Z" });
  });
});

// Item 2 (optimistic hide/pin): a pure handler test proving the optimistic cache patch is
// applied BEFORE the POST resolves, and rolled back on failure — using a real QueryClient
// (no React, no jsdom), same convention as src/lib/api.test.ts.
describe("postPrefsPatch (item 2, no jsdom)", () => {
  it("patches the hub query cache before the POST resolves, then invalidates on success", async () => {
    const client = new QueryClient();
    client.setQueryData(HUB_QUERY_KEY, hubFixture({ p1: { pinned: false, hiddenAt: null } }));
    let resolvePost!: (v: Envelope<ProjectPrefs>) => void;
    const post = () => new Promise<Envelope<ProjectPrefs>>((resolve) => { resolvePost = resolve; });
    const promise = postPrefsPatch(client, "p1", { pinned: true }, post);

    // Optimistic: applied synchronously, before the POST has settled at all.
    const midFlight = client.getQueryData<Envelope<HubEnvelopeData>>(HUB_QUERY_KEY);
    expect(midFlight).toEqual(hubFixture({ p1: { pinned: true, hiddenAt: null } }));

    resolvePost({ ok: true, data: { pinned: true, hiddenAt: null } });
    await promise;
    expect(client.getQueryData<Envelope<HubEnvelopeData>>(HUB_QUERY_KEY)).toEqual(hubFixture({ p1: { pinned: true, hiddenAt: null } }));
  });

  it("rolls back to the pre-patch snapshot on failure", async () => {
    const client = new QueryClient();
    const snapshot = hubFixture({ p1: { pinned: false, hiddenAt: null } });
    client.setQueryData(HUB_QUERY_KEY, snapshot);
    const post = async (): Promise<Envelope<ProjectPrefs>> => ({ ok: false, error: "network error" });
    const result = await postPrefsPatch(client, "p1", { hidden: true }, post);
    expect(result).toEqual({ ok: false, error: "network error" });
    expect(client.getQueryData(HUB_QUERY_KEY)).toEqual(snapshot);
  });

  it("still posts when the hub query hasn't resolved yet, without touching the (empty) cache", async () => {
    const client = new QueryClient();
    const post = async (): Promise<Envelope<ProjectPrefs>> => ({ ok: true, data: { pinned: true, hiddenAt: null } });
    const result = await postPrefsPatch(client, "p1", { pinned: true }, post);
    expect(result).toEqual({ ok: true, data: { pinned: true, hiddenAt: null } });
    expect(client.getQueryData(HUB_QUERY_KEY)).toBeUndefined();
  });
});
