import { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { afterEach, describe, expect, it, vi } from "vitest";
import { agentsOf } from "@/lib/settingsQuery";
import type { WriteOutcome } from "@/components/kanban/PropertiesSidebar";
import type { RulesData } from "@/server/collectors/rules";

const harness = vi.hoisted(() => ({
  query: { data: undefined as unknown, isError: false, dataUpdatedAt: 0, error: undefined as unknown },
  settings: { data: undefined as unknown },
}));

vi.mock("next-intl", () => ({
  useTranslations: () => (key: string) => key,
  useLocale: () => "en-US",
}));
vi.mock("@tanstack/react-query", () => ({
  useQuery: (opts?: { queryKey?: unknown[] }) => (opts?.queryKey?.[0] === "settings" ? harness.settings : harness.query),
  useQueryClient: () => ({ invalidateQueries: vi.fn() }),
}));
vi.mock("@/components/kanban/PropertiesSidebar", async (importOriginal) => {
  const actual = await importOriginal<typeof import("@/components/kanban/PropertiesSidebar")>();
  // keep the real writeGuidance classifier; only the network write is stubbed
  return { ...actual, write: vi.fn().mockResolvedValue({ ok: true }) };
});
vi.mock("./RuleDiffModal", () => ({ RuleDiffModal: () => null }));

import {
  RulesSection,
  captureApplyPreview,
  previewAfterAgents,
  extendEditRange,
  isDraftDirty,
  isDraftStale,
  narrowApplyRevisions,
  ruleStatusForOp,
  savedRevisionOf,
  savedSourceOf,
  seedEditRange,
  settleRuleWrite,
  spliceDraft,
  submittedSource,
  acceptConfirmedBaseline,
  type RuleDraft,
  type RuleStatus,
} from "./RulesSection";
import { ruleOutline } from "@/lib/rules";

const REV = {
  global: "a".repeat(64),
  claude: "b".repeat(64),
  codex: "c".repeat(64),
  opencode: "d".repeat(64),
};

const CANON = "# Title\n\npreamble\n\n## Alpha\nalpha body\n\n## Beta\nbeta body\n";

function makeData(overrides: Partial<RulesData> = {}): RulesData {
  return {
    canonical: CANON,
    exceptions: { claude: "", codex: "", opencode: "" },
    apps: {
      claude: { path: "~/.claude/CLAUDE.md", status: "synced", disk: CANON, diskHash: "h1" },
      codex: { path: "~/.codex/AGENTS.md", status: "drifted", disk: "x", diskHash: "h2" },
      opencode: { path: "~/.config/opencode/AGENTS.md", status: "missing", disk: null, diskHash: "absent" },
    },
    sourceRevisions: REV,
    ...overrides,
  };
}

describe("RulesSection pure derivations", () => {
  it("saved source and revision read the stored slot, not a draft", () => {
    const data = makeData();
    expect(savedSourceOf(data, "global")).toBe(CANON);
    expect(savedSourceOf(data, "claude")).toBe("");
    expect(savedRevisionOf(data, "codex")).toBe(REV.codex);
  });

  it("dirty is text ≠ base; a source revision change under a dirty draft is stale", () => {
    const clean: RuleDraft = { base: "a", baseRevision: REV.global, text: "a" };
    const dirty: RuleDraft = { base: "a", baseRevision: REV.global, text: "b" };
    expect(isDraftDirty(clean)).toBe(false);
    expect(isDraftDirty(dirty)).toBe(true);
    const data = makeData();
    expect(isDraftStale(data, "global", dirty)).toBe(false);
    const changed = makeData({ sourceRevisions: { ...REV, global: "f".repeat(64) } });
    expect(isDraftStale(changed, "global", dirty)).toBe(true);
    expect(isDraftStale(changed, "global", clean)).toBe(false);
  });

  it("narrowApplyRevisions maps the four GET slots to the two-key apply pair", () => {
    expect(narrowApplyRevisions(makeData(), "codex")).toEqual({ global: REV.global, exception: REV.codex });
  });

  it("captureApplyPreview fixes saved content + revisions and distinguishes missing/empty/unreadable", () => {
    const data = makeData({
      exceptions: { claude: "exc\n", codex: "", opencode: "" },
      apps: {
        claude: { path: "/c", status: "drifted", disk: "", diskHash: "emptyhash" },
        codex: { path: "/x", status: "missing", disk: null, diskHash: "absent" },
        opencode: { path: "/o", error: "EACCES" },
      },
    });
    const empty = captureApplyPreview(data, "claude");
    expect(empty).toEqual({
      app: "claude",
      canonical: CANON,
      exception: "exc\n",
      revisions: { global: REV.global, exception: REV.claude },
      disk: "",
      diskHash: "emptyhash",
      missing: false,
      path: "/c",
    });
    const missing = captureApplyPreview(data, "codex");
    expect(missing).toMatchObject({ disk: "", diskHash: "absent", missing: true });
    expect(captureApplyPreview(data, "opencode")).toBeNull(); // unreadable → no empty-file preview
  });
});

describe("RulesSection — SSR", () => {
  it("renders the master/detail outline, process strip and destination cards", () => {
    harness.query = { data: { ok: true, data: makeData() }, isError: false, dataUpdatedAt: 0, error: undefined };
    const html = renderToStaticMarkup(createElement(RulesSection));
    expect(html).toContain("groupGlobal");
    expect(html).toContain("groupExceptions");
    expect(html).toContain("processEdit");
    expect(html).toContain("Alpha");
    expect(html).toContain("Beta");
    expect(html).toContain("destHeading");
    expect(html).toContain("~/.config/opencode/AGENTS.md");
  });

  it("falls back to whole-document only when the source has no H2 outline", () => {
    harness.query = {
      data: { ok: true, data: makeData({ canonical: "just a paragraph\nno headings here\n" }) },
      isError: false,
      dataUpdatedAt: 0,
      error: undefined,
    };
    const html = renderToStaticMarkup(createElement(RulesSection));
    expect(html).toContain("sectionTitle");
    expect(html).not.toContain("Alpha");
  });

  it("keeps last good rules under a visible warning", () => {
    harness.query = { data: { ok: true, data: makeData() }, isError: true, dataUpdatedAt: 4, error: new Error("down") };
    const html = renderToStaticMarkup(createElement(RulesSection));
    expect(html).toContain("unavailable");
    expect(html).toContain("destHeading");
  });

  it("renders the unavailable heading at weight 900 (font-black), not 700, now that the display font is variable", () => {
    harness.query = { data: undefined, isError: true, dataUpdatedAt: 0, error: new Error("down") };
    const html = renderToStaticMarkup(createElement(RulesSection));
    expect(html).toContain("sectionTitle");
    expect(html).toContain("font-display text-[15px] font-black");
    expect(html).not.toContain("font-bold");
  });
});

describe("settleRuleWrite (production rule settlement)", () => {
  const t = (key: string) => key;

  it("reports pending then success and calls onSuccess exactly once on a confirmed ok", async () => {
    const status: RuleStatus[] = [];
    const onSuccess = vi.fn();
    const result = await settleRuleWrite({
      op: "edit:global",
      post: async () => ({ ok: true }),
      t,
      pendingText: "saving",
      successText: "saved",
      onStatus: (s) => status.push(s),
      onSuccess,
    });
    expect(result).toBeNull();
    expect(status).toEqual([
      { tone: "pending", text: "saving", op: "edit:global" },
      { tone: "success", text: "saved", op: "edit:global" },
    ]);
    expect(onSuccess).toHaveBeenCalledTimes(1);
  });

  it("maps refused, unconfirmed and applied-unrecorded to an error status and never calls onSuccess", async () => {
    const cases: Array<{ outcome: WriteOutcome; expectedText: string }> = [
      { outcome: { ok: false, error: "invalid payload" }, expectedText: "writeRefused" },
      {
        outcome: { ok: false, code: "mutation-unconfirmed", effect: "unconfirmed", audit: "recorded", error: "t" },
        expectedText: "writeUnconfirmed",
      },
      {
        outcome: { ok: false, code: "audit-finalization-failed", effect: "applied", audit: "pending", error: "t" },
        expectedText: "writeAppliedUnrecorded",
      },
    ];
    for (const { outcome, expectedText } of cases) {
      const status: RuleStatus[] = [];
      const onSuccess = vi.fn();
      const result = await settleRuleWrite({
        op: "apply:claude",
        post: async () => outcome,
        t,
        pendingText: "applying",
        successText: "applied",
        onStatus: (s) => status.push(s),
        onSuccess,
      });
      expect(result).toBe(expectedText);
      expect(status).toEqual([
        { tone: "pending", text: "applying", op: "apply:claude" },
        { tone: "error", text: expectedText, op: "apply:claude" },
      ]);
      expect(onSuccess).not.toHaveBeenCalled();
    }
  });
});

describe("RulesSection editing range (F7 regression)", () => {
  it("keeps a stable range while a literal new H2 heading is typed inside section A", () => {
    const base = "## A\nalpha\n\n## B\nbeta\n";
    const outline = ruleOutline(base);
    expect(outline.spans.map((s) => s.label)).toEqual(["A", "B"]);

    // selection time: capture section A's exact range/buffer
    let range = seedEditRange(base, 0)!;
    const typed1 = "alpha\n## new\n";
    const draft1 = spliceDraft(base, range, typed1);
    range = extendEditRange(range, typed1);
    expect(draft1).toBe("alpha\n## new\n## B\nbeta\n");

    // the outline now has a different span set; continuing to type must NOT
    // relocate the buffer (splice against the range captured one keystroke ago)
    expect(ruleOutline(draft1).spans.map((s) => s.label)).toEqual(["new", "B"]);
    const typed2 = `${typed1}more\n`;
    const draft2 = spliceDraft(draft1, range, typed2);
    range = extendEditRange(range, typed2);
    expect(draft2).toBe("alpha\n## new\nmore\n## B\nbeta\n");
    // the other section is untouched and still intact
    expect(draft2).toContain("## B\nbeta\n");
    expect(ruleOutline(draft2).spans.map((s) => s.label)).toEqual(["new", "B"]);
  });
});

describe("RulesSection normalized source save", () => {
  it("submits one trailing newline and accepts the normalized GET as the baseline", () => {
    const draft = "just a paragraph without a final newline";
    const submitted = submittedSource("global", draft);
    expect(submitted).toBe("just a paragraph without a final newline\n");
    // the server stores normalized content: an unchanged editor accepts it,
    // comparing the (raw) draft against the RAW submitted text, not the
    // normalized server bytes
    expect(acceptConfirmedBaseline(submitted, submitted, draft, draft)).toBe("accept");
    // a newer local edit is retained, not overwritten by the confirmed save
    expect(acceptConfirmedBaseline(submitted, submitted, draft, `${submitted}more`)).toBe("keep-newer");
    // any other source content is a visible conflict, never an inferred revision
    expect(acceptConfirmedBaseline("different\n", submitted, draft, draft)).toBe("conflict");
    // a normalized save whose editor still holds the raw draft is accepted, not
    // misread as a newer edit (the regression the browser probe caught)
    expect(acceptConfirmedBaseline(submitted, submitted, draft, draft)).not.toBe("keep-newer");
  });

  it("keeps the empty-exception sentinel unnormalized", () => {
    expect(submittedSource("claude", "")).toBe("");
    expect(submittedSource("claude", "exc\n\n\n")).toBe("exc\n");
  });
});

describe("ruleStatusForOp (dialog-scoped operation identity)", () => {
  it("surfaces only the settlement of the matching operation", () => {
    const priorApplyError = { tone: "error" as const, text: "writeRefused", op: "apply:claude" as const };
    expect(ruleStatusForOp(priorApplyError, "edit:global")).toBeNull();
    expect(ruleStatusForOp(priorApplyError, "edit:claude")).toBeNull();
    expect(ruleStatusForOp(priorApplyError, "apply:opencode")).toBeNull();
    expect(ruleStatusForOp(priorApplyError, "apply:claude")).toEqual(priorApplyError);
    expect(ruleStatusForOp(null, "apply:claude")).toBeNull();
  });
});

describe("RulesSection — agent switches (MOA-504 D10)", () => {
  afterEach(() => { harness.settings = { data: undefined }; });
  const withAgents = (claude: boolean, codex: boolean, opencode: boolean) =>
    ({ data: { ok: true, data: { integrations: { agents: { claude, codex, opencode } } } } });

  it("hides the exception slot AND the destination card of an agent that is off", () => {
    harness.query = { data: { ok: true, data: makeData() }, isError: false, dataUpdatedAt: 0, error: undefined };
    harness.settings = withAgents(false, true, true);
    const html = renderToStaticMarkup(createElement(RulesSection));
    expect(html).not.toContain("apps.claude");
    expect(html).toContain("apps.codex");
    expect(html).toContain("apps.opencode");
  });

  it("fails open while the settings are unknown: all three agents are shown", () => {
    harness.query = { data: { ok: true, data: makeData() }, isError: false, dataUpdatedAt: 0, error: undefined };
    harness.settings = { data: undefined };
    const html = renderToStaticMarkup(createElement(RulesSection));
    for (const app of ["claude", "codex", "opencode"]) expect(html).toContain(`apps.${app}`);
  });
});

describe("RulesSection — an open apply preview closes when its agent is switched off (MOA-504 D10)", () => {
  const result = (claude: boolean, codex: boolean) =>
    ({ ok: true, data: { integrations: { agents: { claude, codex, opencode: true } } } }) as never;
  const preview = captureApplyPreview(makeData(), "claude")!;

  it("keeps the preview while its agent is on, and drops it the moment the settings result disables that agent", () => {
    expect(preview).not.toBeNull();
    expect(previewAfterAgents(preview, agentsOf(result(true, true)))).toBe(preview);
    expect(previewAfterAgents(preview, agentsOf(result(false, true)))).toBeNull();
  });
  it("is untouched by ANOTHER agent being switched off, by no preview, and by unknown settings (fail open)", () => {
    expect(previewAfterAgents(preview, agentsOf(result(true, false)))).toBe(preview);
    expect(previewAfterAgents(null, agentsOf(result(false, false)))).toBeNull();
    expect(previewAfterAgents(preview, agentsOf(undefined))).toBe(preview);
  });
});
