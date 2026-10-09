import { execFile } from "node:child_process";
import { promisify } from "node:util";
import { collectorResponse } from "../../../../server/api";
import { capturePaneTail, listLivePanes } from "../../../../server/collectors/workflow-tmux";

type ExecFile = (file: string, args: string[]) => Promise<string>;

const execFileAsync = promisify(execFile);
// Spec §7 Decision 12: this capture's OWN 1s exec timeout — distinct from workflow-tmux.ts's own
// systemExec (5s default, used by listLivePanes/injectAnswers/injectFreeform). The ROUTE decides
// which `run` implementation capturePaneTail gets, never the collector function itself.
const captureExec: ExecFile = async (file, args) => {
  const { stdout } = await execFileAsync(file, args, { encoding: "utf8", timeout: 1_000, maxBuffer: 1024 * 1024 });
  return stdout;
};

const MAX_PANES = 40;
const LINES_PER_PANE = 8;
const TOTAL_DEADLINE_MS = 2_500;

export type PanesTailRouteDeps = {
  listLivePanes: typeof listLivePanes;
  capturePaneTail: typeof capturePaneTail;
  run: ExecFile;
  // Injected (not a hardcoded module constant) so tests can shrink it to prove the deadline is
  // actually enforced without a real 2.5s wait — systemDeps below uses the real value.
  deadlineMs: number;
};

const systemDeps: PanesTailRouteDeps = {
  listLivePanes, capturePaneTail, run: captureExec, deadlineMs: TOTAL_DEADLINE_MS,
};

// Races one capture against a REMAINING-time budget — never lets one slow/hung pane block a
// response the other panes already finished (Decision 12).
function withDeadline<T>(p: Promise<T>, ms: number): Promise<T> {
  return new Promise((resolve, reject) => {
    const timer = setTimeout(() => reject(new Error("capture deadline exceeded")), ms);
    p.then(
      (v) => { clearTimeout(timer); resolve(v); },
      (e) => { clearTimeout(timer); reject(e); },
    );
  });
}

// Spec §7 Decisions 9-12: no query param (Decision 11 — the route always requests
// LINES_PER_PANE=8; Part 2's desktop cell slices to 4 client-side). {ok:false} ONLY when
// listLivePanes() itself fails (tmux unavailable) — never for one pane's capture. Every capture
// runs CONCURRENTLY (Promise.allSettled). Round-2 review F2: ONE absolute batch deadline
// (`deadlineAt`) is computed once, here, for the whole batch — not a fresh `deadlineMs`-long timer
// per capture, which would let a capture that starts even a tick later than its siblings get a
// slightly longer runway. Each capture races `withDeadline` against `max(0, deadlineAt - Date.now())`
// — the time actually left on the shared clock at the moment that capture starts — so panes started
// together (they are: `.map` over `panes` runs synchronously, before any `await`) get effectively
// the same remaining budget. A rejected/timed-out one maps to lines:null (Decision 11), every other
// pane's array is untouched.
export async function handlePanesTailGet(deps: PanesTailRouteDeps = systemDeps) {
  return collectorResponse<{ pane: string; lines: string[] | null }[]>(async () => {
    const live = await deps.listLivePanes();
    const panes = [...live.paneIds].slice(0, MAX_PANES);
    const deadlineAt = Date.now() + deps.deadlineMs;
    const results = await Promise.allSettled(
      panes.map((pane) => withDeadline(deps.capturePaneTail(pane, LINES_PER_PANE, deps.run), Math.max(0, deadlineAt - Date.now()))),
    );
    return panes.map((pane, i) => {
      const r = results[i];
      return { pane, lines: r.status === "fulfilled" ? r.value : null };
    });
  });
}
