import { readdirSync, readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { describe, expect, it, test } from "vitest";
import { openDb } from "./index";
import { insertEvent, getHubData, type HubLastRun, type LiveSnapshot } from "./workflows";
import { parseIngress } from "../collectors/workflow-events";

// src/server/db -> ../../../scripts/fixtures/moa-474
const FIXTURES_DIR = join(
  dirname(fileURLToPath(import.meta.url)), "../../../scripts/fixtures/moa-474",
);
const LIVE: LiveSnapshot = { paneCommands: new Map(), paneIds: new Set(), incarnation: null };

type Fixture = {
  runStarted: Record<string, unknown>;
  runFinished: Record<string, unknown>;
  expected: Partial<HubLastRun>;
};

function loadFixtures(): [string, Fixture][] {
  let names: string[];
  try {
    names = readdirSync(FIXTURES_DIR).filter((f) => f.endsWith(".json"));
  } catch {
    return [];
  }
  return names.map((name) => [name, JSON.parse(readFileSync(join(FIXTURES_DIR, name), "utf-8")) as Fixture]);
}

const fixtures = loadFixtures();

describe("MOA-474 Part A cross-language boundary (Python producer -> TS ingress -> DB)", () => {
  // This task only proves the Python producer's runStarted/runFinished envelopes pass
  // real ingress and land in the DB unchanged. The read-back through
  // lastRunFor/getHubData's projection (the "expected" block) is the second describe
  // below, over the same fixtures.
  const files = readdirSync(FIXTURES_DIR).filter((f) => f.endsWith(".json"));

  test("the fixture directory is not empty", () => {
    expect(files.length).toBeGreaterThan(0);
  });

  for (const file of files) {
    test(`${file}: runStarted and runFinished both pass ingress and persist unchanged`, () => {
      const { runStarted, runFinished } = JSON.parse(readFileSync(join(FIXTURES_DIR, file), "utf-8"));
      const db = openDb(":memory:");
      try {
        const startedResult = parseIngress(runStarted);
        expect(startedResult.ok, `${file} runStarted`).toBe(true);
        if (!startedResult.ok) return;
        const startedRow = insertEvent(db, startedResult.event);
        expect(startedRow.payload).toEqual(runStarted.payload);

        const finishedResult = parseIngress(runFinished);
        expect(finishedResult.ok, `${file} runFinished`).toBe(true);
        if (!finishedResult.ok) return;
        const finishedRow = insertEvent(db, finishedResult.event);
        expect(finishedRow.payload).toEqual(runFinished.payload);
        expect(finishedRow.run_id).toBe(runFinished.run_id);
      } finally {
        db.close();
      }
    });
  }
});

describe("MOA-474 cross-language boundary — Part A fixtures read back through lastRunFor", () => {
  // Fixed set, one per representative decision-table row -- must match Part A2 Task 3's
  // `_write_fixture` names exactly; a missing or extra file fails here, not silently.
  const EXPECTED_FIXTURES = [
    "builder-ok-success.json",
    "builder-runtime-failure.json",
    "builder-verify-failure.json",
    "builder-interrupted-worker.json",
    "reviewer-no-verdict.json",
    "reviewer-ok-approve.json",
    "builder-summary-200-non-bmp-emoji.json",
    "cancelled-row-unchanged-shape.json",
  ];
  it("the fixture directory holds exactly the Part A2 set", () => {
    expect(fixtures.map(([name]) => name).sort()).toEqual([...EXPECTED_FIXTURES].sort());
  });

  it.each(fixtures)("%s round-trips through parseIngress -> insertEvent -> lastRunFor", (_name, fixture) => {
    const db = openDb(":memory:");
    try {
      const started = parseIngress(fixture.runStarted);
      expect(started.ok).toBe(true);
      if (started.ok) insertEvent(db, started.event, new Date());

      const finished = parseIngress(fixture.runFinished);
      expect(finished.ok).toBe(true);
      if (!finished.ok) return;
      insertEvent(db, finished.event, new Date(Date.now() + 1000));

      const project = (fixture.runFinished as { project: string }).project;
      const lastRun = getHubData(db, LIVE, {}).byProject[project]?.lastRun;
      expect(lastRun).toMatchObject(fixture.expected);
    } finally {
      db.close();
    }
  });
});
