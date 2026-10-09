import { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { afterEach, describe, expect, it, vi } from "vitest";
import type { Envelope } from "@/lib/api";
import en from "../../../messages/en-US.json";
import pt from "../../../messages/pt-BR.json";
import { QueryClient } from "@tanstack/react-query";
import type { EventType } from "@/lib/workflow";

type Settings = { archiveAfterDays: number };

type AfkSettings = { enabled: boolean; forwardTypes: EventType[] };

const harness = vi.hoisted(() => ({
  archiveQuery: { data: undefined as Envelope<Settings> | undefined, isError: false },
  afkQuery: { data: undefined as Envelope<AfkSettings> | undefined, isError: false },
}));

vi.mock("next-intl", () => ({ useTranslations: () => (key: string) => key }));
vi.mock("@tanstack/react-query", async (importOriginal) => {
  const actual = await importOriginal<typeof import("@tanstack/react-query")>();
  return {
    ...actual,
    useQuery: ({ queryKey }: { queryKey: unknown[] }) => (queryKey[1] === "afk" ? harness.afkQuery : harness.archiveQuery),
    useQueryClient: () => ({ setQueryData: vi.fn() }),
  };
});

import { AFK_QUERY_KEY, computeForwardTypesToggle, ForwardTypesFailedCaption, handleForwardTypeToggle, MissionSettingsSection, postArchiveAfterDays, postForwardTypes } from "./MissionSettingsSection";

describe("MissionSettingsSection", () => {
  it("renders a number input, disabled until the GET resolves", () => {
    harness.archiveQuery = { data: undefined, isError: false };
    harness.afkQuery = { data: undefined, isError: false };
    const html = renderToStaticMarkup(createElement(MissionSettingsSection));
    expect(html).toContain('type="number"');
    expect(html).toContain("disabled");
  });

  it("shows the loaded value once the GET resolves", () => {
    harness.archiveQuery = { data: { ok: true, data: { archiveAfterDays: 21 } }, isError: false };
    harness.afkQuery = { data: undefined, isError: false };
    expect(renderToStaticMarkup(createElement(MissionSettingsSection))).toContain('value="21"');
  });
});

describe("postArchiveAfterDays (plan review round 1 F2 — POST call, fetch spy args)", () => {
  afterEach(() => { vi.restoreAllMocks(); });

  it("POSTs the bounded value to /api/mission/settings with the exact body", async () => {
    const spy = vi.spyOn(global, "fetch").mockResolvedValue({ json: async () => ({ ok: true, data: { archiveAfterDays: 10 } }) } as Response);
    const res = await postArchiveAfterDays(10);
    expect(spy).toHaveBeenCalledWith("/api/mission/settings", {
      method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ archiveAfterDays: 10 }),
    });
    expect(res).toEqual({ ok: true, data: { archiveAfterDays: 10 } });
  });
});

describe("forward-types checklist (MOA-487 §5 Decision 6/7)", () => {
  it("renders exactly six checkboxes, pre-checked per §5 Decision 5 defaults, once the GET resolves", () => {
    harness.archiveQuery = { data: undefined, isError: false };
    harness.afkQuery = { data: { ok: true, data: { enabled: true, forwardTypes: ["question", "attention-needed", "mission-finished"] } }, isError: false };
    const html = renderToStaticMarkup(createElement(MissionSettingsSection));
    const rows = [...html.matchAll(/<label[^>]*>(<input[^>]*>)([^<]*)<\/label>/g)].map((m) => ({ input: m[1], text: m[2] }));
    expect(rows).toHaveLength(6);
    const checkedFor = (labelKey: string) => rows.find((r) => r.text === labelKey)?.input.includes("checked") ?? false;
    expect(checkedFor("forwardTypeQuestion")).toBe(true);
    expect(checkedFor("forwardTypeAttentionNeeded")).toBe(true);
    expect(checkedFor("forwardTypeMissionFinished")).toBe(true);
    expect(checkedFor("forwardTypeRunFinished")).toBe(false);
    expect(checkedFor("forwardTypeMergeApproved")).toBe(false);
    expect(checkedFor("forwardTypeMissionStatusUpdated")).toBe(false);
  });

  it("carries data-forward-type on every checkbox so the toggle wiring is visible (review F3)", () => {
    harness.archiveQuery = { data: undefined, isError: false };
    harness.afkQuery = { data: { ok: true, data: { enabled: true, forwardTypes: ["question"] } }, isError: false };
    const html = renderToStaticMarkup(createElement(MissionSettingsSection));
    for (const type of ["question", "attention-needed", "mission-finished", "run-finished", "merge-approved", "mission-status-updated"]) {
      expect(html).toContain(`data-forward-type="${type}"`);
    }
  });

  it("disables every checkbox while a save is in flight (review F2)", () => {
    harness.archiveQuery = { data: undefined, isError: false };
    harness.afkQuery = { data: { ok: true, data: { enabled: true, forwardTypes: ["question"] } }, isError: false };
    const html = renderToStaticMarkup(createElement(MissionSettingsSection, { initialForwardSaving: true }));
    const checkboxes = [...html.matchAll(/<input type="checkbox"[^>]*>/g)].map((m) => m[0]);
    expect(checkboxes).toHaveLength(6);
    for (const checkbox of checkboxes) expect(checkbox).toContain("disabled");
  });

  it("every new message key exists as a string in both locales (AGENTS.md rule 4)", () => {
    const keys = [
      "forwardTypesTitle", "forwardTypeQuestion", "forwardTypeAttentionNeeded", "forwardTypeMissionFinished",
      "forwardTypeRunFinished", "forwardTypeMergeApproved", "forwardTypeMissionStatusUpdated", "forwardTypesSaveFailed",
    ] as const;
    for (const key of keys) {
      expect(typeof (pt.tools as Record<string, unknown>)[key]).toBe("string");
      expect(typeof (en.tools as Record<string, unknown>)[key]).toBe("string");
    }
  });
});

describe("computeForwardTypesToggle (pure add/remove, no duplicates)", () => {
  it("adds the type when checked, removes when unchecked, is a no-op when redundant", () => {
    expect(computeForwardTypesToggle(["question"], "attention-needed", true)).toEqual(["question", "attention-needed"]);
    expect(computeForwardTypesToggle(["question", "attention-needed"], "question", false)).toEqual(["attention-needed"]);
    expect(computeForwardTypesToggle(["question"], "question", true)).toEqual(["question"]);
    expect(computeForwardTypesToggle(["question"], "run-finished", false)).toEqual(["question"]);
  });
});

describe("postForwardTypes (helper-level, mirrors postPrefsPatch's own convention — plan review round 2 F2/F3, no jsdom)", () => {
  afterEach(() => { vi.restoreAllMocks(); });

  it("patches the afk query cache before the POST resolves, then invalidates and resolves true on success", async () => {
    const client = new QueryClient();
    client.setQueryData(AFK_QUERY_KEY, { ok: true, data: { enabled: true, forwardTypes: ["question"] } });
    let resolvePost!: (v: Envelope<AfkSettings>) => void;
    const post = () => new Promise<Envelope<AfkSettings>>((resolve) => { resolvePost = resolve; });
    const promise = postForwardTypes(client, ["question", "attention-needed"], post);

    // Optimistic: applied synchronously, before the POST has settled at all.
    expect(client.getQueryData(AFK_QUERY_KEY)).toEqual({ ok: true, data: { enabled: true, forwardTypes: ["question", "attention-needed"] } });

    resolvePost({ ok: true, data: { enabled: true, forwardTypes: ["question", "attention-needed"] } });
    expect(await promise).toBe(true);
    expect(client.getQueryData(AFK_QUERY_KEY)).toEqual({ ok: true, data: { enabled: true, forwardTypes: ["question", "attention-needed"] } });
  });

  it("rolls back to the pre-patch snapshot and resolves false on failure", async () => {
    const client = new QueryClient();
    const snapshot = { ok: true as const, data: { enabled: true, forwardTypes: ["question"] } };
    client.setQueryData(AFK_QUERY_KEY, snapshot);
    const post = async (): Promise<Envelope<AfkSettings>> => ({ ok: false, error: "network error" });
    expect(await postForwardTypes(client, ["question", "attention-needed"], post)).toBe(false);
    expect(client.getQueryData(AFK_QUERY_KEY)).toEqual(snapshot);
  });

  it("rolls back to the pre-patch snapshot and resolves false when post rejects (review F1)", async () => {
    const client = new QueryClient();
    const snapshot = { ok: true as const, data: { enabled: true, forwardTypes: ["question"] } };
    client.setQueryData(AFK_QUERY_KEY, snapshot);
    const post = async (): Promise<Envelope<AfkSettings>> => { throw new Error("network error"); };
    expect(await postForwardTypes(client, ["question", "attention-needed"], post)).toBe(false);
    expect(client.getQueryData(AFK_QUERY_KEY)).toEqual(snapshot);
  });
});

describe("handleForwardTypeToggle (review F3 — the actual checkbox handler, tested directly)", () => {
  afterEach(() => { vi.restoreAllMocks(); });

  it("POSTs the toggled array and patches the cache optimistically before the POST resolves", async () => {
    const client = new QueryClient();
    client.setQueryData(AFK_QUERY_KEY, { ok: true, data: { enabled: true, forwardTypes: ["question"] } });
    let resolvePost!: (v: Envelope<AfkSettings>) => void;
    const post = vi.fn(() => new Promise<Envelope<AfkSettings>>((resolve) => { resolvePost = resolve; }));
    const setSaving = vi.fn();
    const setFailed = vi.fn();

    const promise = handleForwardTypeToggle(
      { queryClient: client, current: ["question"], type: "attention-needed", checked: true, post },
      setSaving, setFailed,
    );

    expect(setSaving).toHaveBeenCalledWith(true);
    expect(post).toHaveBeenCalledWith("/api/workflow/afk", { forwardTypes: ["question", "attention-needed"] });
    expect(client.getQueryData(AFK_QUERY_KEY)).toEqual({ ok: true, data: { enabled: true, forwardTypes: ["question", "attention-needed"] } });

    resolvePost({ ok: true, data: { enabled: true, forwardTypes: ["question", "attention-needed"] } });
    await promise;
    expect(setFailed).toHaveBeenCalledWith(false);
    expect(setSaving).toHaveBeenCalledWith(false);
  });

  it("rolls back and sets the failed flag on {ok:false}", async () => {
    const client = new QueryClient();
    const snapshot = { ok: true as const, data: { enabled: true, forwardTypes: ["question"] } };
    client.setQueryData(AFK_QUERY_KEY, snapshot);
    const post = async (): Promise<Envelope<AfkSettings>> => ({ ok: false, error: "network error" });
    const setSaving = vi.fn();
    const setFailed = vi.fn();

    await handleForwardTypeToggle({ queryClient: client, current: ["question"], type: "run-finished", checked: true, post }, setSaving, setFailed);

    expect(client.getQueryData(AFK_QUERY_KEY)).toEqual(snapshot);
    expect(setFailed).toHaveBeenCalledWith(true);
    expect(setSaving).toHaveBeenNthCalledWith(1, true);
    expect(setSaving).toHaveBeenNthCalledWith(2, false);
  });

  it("rolls back and sets the failed flag when post rejects", async () => {
    const client = new QueryClient();
    const snapshot = { ok: true as const, data: { enabled: true, forwardTypes: ["question"] } };
    client.setQueryData(AFK_QUERY_KEY, snapshot);
    const post = async (): Promise<Envelope<AfkSettings>> => { throw new Error("network error"); };
    const setSaving = vi.fn();
    const setFailed = vi.fn();

    await handleForwardTypeToggle({ queryClient: client, current: ["question"], type: "run-finished", checked: true, post }, setSaving, setFailed);

    expect(client.getQueryData(AFK_QUERY_KEY)).toEqual(snapshot);
    expect(setFailed).toHaveBeenCalledWith(true);
    expect(setSaving).toHaveBeenNthCalledWith(2, false);
  });
});

describe("ForwardTypesFailedCaption (plan review round 2 F3 — isolates the flag's conditional render)", () => {
  it("renders the caption when failed, nothing when not", () => {
    expect(renderToStaticMarkup(createElement(ForwardTypesFailedCaption, { failed: true, t: (k: string) => k }))).toContain("forwardTypesSaveFailed");
    expect(renderToStaticMarkup(createElement(ForwardTypesFailedCaption, { failed: false, t: (k: string) => k }))).toBe("");
  });
});
