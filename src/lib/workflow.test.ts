import { describe, expect, it } from "vitest";
import {
  CALLERS, CAPSULE_RULES, DEFAULT_FORWARD_TYPES, EVENT_MATRIX, EVENT_TYPES, FORWARDABLE_EVENT_TYPES, GITHUB_PR_URL_RE, LIMITS, MISSION_MILESTONE_NUMERIC_RE, RUN_ID_RE,
  validMissionText, validOptionalFocus, validPhaseToken, validRefName, validSingleLineCommand,
} from "./workflow";

describe("Phase 2 shared vocabulary", () => {
  it("CAPSULE_RULES includes classifier-off (spec MOA-502 Decision 1)", () => {
    expect(CAPSULE_RULES).toContain("classifier-off");
  });

  it("CAPSULE_RULES includes merge-question (merge question contract)", () => {
    expect(CAPSULE_RULES).toContain("merge-question");
  });

  it("CALLERS gains jaxos; RUN_ID_RE is a 12-hex jaxflow run id", () => {
    expect(CALLERS).toEqual(["claude", "codex", "jaxos"]);
    expect(RUN_ID_RE.test("a1b2c3d4e5f6")).toBe(true);
    expect(RUN_ID_RE.test("A1B2C3D4E5F6")).toBe(false);
    expect(RUN_ID_RE.test("a1b2c3d4e5f")).toBe(false);
    expect(LIMITS.checks).toBe(2000);
    expect(LIMITS.focus).toBe(2000);
    expect(LIMITS.whitelist).toBe(4096);
  });

  it("validRefName: ≤512 units, no whitespace, no NUL", () => {
    expect(validRefName("feat/x")).toBe(true);
    expect(validRefName("")).toBe(false);
    expect(validRefName("feat x")).toBe(false);
    expect(validRefName("a".repeat(513))).toBe(false);
    expect(validRefName("-x")).toBe(true);
    expect(validRefName("a\0b")).toBe(false); // round-2 F1: NUL breaks argv, not just a control char
  });

  it("validSingleLineCommand: non-blank, single line, capped", () => {
    expect(validSingleLineCommand("pnpm test", LIMITS.verify)).toBe(true);
    expect(validSingleLineCommand("   ", LIMITS.verify)).toBe(false);
    expect(validSingleLineCommand("pnpm test\npnpm build", LIMITS.verify)).toBe(false);
    expect(validSingleLineCommand("pnpm test\r", LIMITS.verify)).toBe(false);
    expect(validSingleLineCommand("pnpm\ttest", LIMITS.verify)).toBe(false);
    expect(validSingleLineCommand("pnpm test\0", LIMITS.verify)).toBe(false);
    expect(validSingleLineCommand("x".repeat(500), LIMITS.verify)).toBe(true);
    expect(validSingleLineCommand("x".repeat(501), LIMITS.verify)).toBe(false);
  });

  it("validOptionalFocus: absent is fine, present must be a single line", () => {
    expect(validOptionalFocus(undefined)).toBe(true);
    expect(validOptionalFocus("look at the guard")).toBe(true);
    expect(validOptionalFocus("")).toBe(false);
    expect(validOptionalFocus("a\nb")).toBe(false);
    expect(validOptionalFocus("x".repeat(2001))).toBe(false);
  });

  it("validPhaseToken: [A-Za-z0-9._-], 1-64", () => {
    expect(validPhaseToken("moa-481-x")).toBe(true);
    expect(validPhaseToken("bad phase")).toBe(false);
    expect(validPhaseToken("")).toBe(false);
    expect(validPhaseToken("a".repeat(65))).toBe(false);
  });
});

describe("FORWARDABLE_EVENT_TYPES / DEFAULT_FORWARD_TYPES (MOA-487 §5 Decisions 2/4/5)", () => {
  it("EVENT_TYPES gained the two reserved mission types, each with a forward-policy matrix row", () => {
    expect(EVENT_TYPES).toContain("mission-status-updated");
    expect(EVENT_TYPES).toContain("mission-finished");
    expect(EVENT_MATRIX["mission-status-updated"]).toMatchObject({ policy: "forward", roles: ["lead"], runScoped: false });
    expect(EVENT_MATRIX["mission-finished"]).toMatchObject({ policy: "forward", roles: ["lead"], runScoped: false });
  });

  it("FORWARDABLE_EVENT_TYPES is exactly the seven forward-policy types, no more, no fewer", () => {
    expect(new Set(FORWARDABLE_EVENT_TYPES)).toEqual(new Set([
      "run-finished", "question", "attention-needed", "merge-approved",
      "mission-status-updated", "mission-finished", "pr-opened",
    ]));
  });

  it("DEFAULT_FORWARD_TYPES is a subset of FORWARDABLE_EVENT_TYPES", () => {
    for (const t of DEFAULT_FORWARD_TYPES) expect(FORWARDABLE_EVENT_TYPES).toContain(t);
    expect(DEFAULT_FORWARD_TYPES).toEqual(["question", "attention-needed", "mission-finished"]);
  });
});

describe("mission field bounds and validators (spec §6a)", () => {
  it("validMissionText requires non-blank, already-trimmed, one line, within max", () => {
    expect(validMissionText("Ship it", 80)).toBe(true);
    expect(validMissionText("", 80)).toBe(false);
    expect(validMissionText(" untrimmed", 80)).toBe(false);
    expect(validMissionText("untrimmed ", 80)).toBe(false);
    expect(validMissionText("a\nb", 80)).toBe(false);
    expect(validMissionText("a\tb", 80)).toBe(false);
    expect(validMissionText("a".repeat(81), 80)).toBe(false);
    expect(validMissionText("a".repeat(80), 80)).toBe(true);
    expect(validMissionText(42, 80)).toBe(false);
  });

  it("MISSION_MILESTONE_NUMERIC_RE matches only a whole positive-integer string (spec §6 Decision 5)", () => {
    expect(MISSION_MILESTONE_NUMERIC_RE.test("3")).toBe(true);
    expect(MISSION_MILESTONE_NUMERIC_RE.test("1")).toBe(true);
    expect(MISSION_MILESTONE_NUMERIC_RE.test("0")).toBe(false);
    expect(MISSION_MILESTONE_NUMERIC_RE.test("03")).toBe(false);
    expect(MISSION_MILESTONE_NUMERIC_RE.test("3a")).toBe(false);
    expect(MISSION_MILESTONE_NUMERIC_RE.test("Ship it")).toBe(false);
  });

  it("LIMITS carries the mission bounds spec §6a/§7 name (80/280/280/60, cap 12)", () => {
    expect(LIMITS.missionName).toBe(80);
    expect(LIMITS.missionGoal).toBe(280);
    expect(LIMITS.missionStatusLine).toBe(280);
    expect(LIMITS.milestoneTitle).toBe(60);
    expect(LIMITS.missionMilestonesMax).toBe(12);
  });
});

describe("pr-opened hub event type (spec MOA-465 Ledger & card)", () => {
  it("EVENT_TYPES/EVENT_MATRIX gained pr-opened as a forward-policy, non-run-scoped, lead-only event", () => {
    expect(EVENT_TYPES).toContain("pr-opened");
    expect(EVENT_MATRIX["pr-opened"]).toEqual({ emitters: ["wrapper"], roles: ["lead"], runScoped: false, policy: "forward" });
  });

  it("LIMITS.prUrl is generous enough for a real GitHub PR URL", () => {
    expect(LIMITS.prUrl).toBe(300);
  });

  it("GITHUB_PR_URL_RE matches only a github.com pull URL with a numeric PR number", () => {
    expect(GITHUB_PR_URL_RE.test("https://github.com/acme/route-converter-se/pull/123")).toBe(true);
    expect(GITHUB_PR_URL_RE.test("https://github.com/acme/route-converter-se/pull/0")).toBe(false);
    expect(GITHUB_PR_URL_RE.test("https://gitlab.com/acme/x/pull/1")).toBe(false);
    expect(GITHUB_PR_URL_RE.test("https://github.com/acme/x/issues/1")).toBe(false);
    expect(GITHUB_PR_URL_RE.test("not a url")).toBe(false);
  });
});
