import { execFile } from "node:child_process";
import { join } from "node:path";
import { promisify } from "node:util";

const execFileAsync = promisify(execFile);

export type ExecFile = (file: string, args: string[]) => Promise<string>;

async function systemExec(file: string, args: string[]): Promise<string> {
  const options = file === "gh"
    ? { encoding: "utf8" as const, timeout: 15_000, maxBuffer: 2 * 1024 * 1024 }
    : { encoding: "utf8" as const, timeout: 5_000, maxBuffer: 1024 * 1024 };
  const { stdout } = await execFileAsync(file, args, options);
  return stdout;
}

export type CiState = "green" | "pending" | "failing" | "none";

export type Pr = {
  number: number;
  title: string;
  isDraft: boolean;
  url: string;
  ciState: CiState;
  mergeReady: boolean;
  // Phase 3 §9 (Decision 13): the ticker's timestamp for an open PR line.
  updatedAt: string;
};

export type PrResult = { ok: true; prs: Pr[]; truncated: boolean } | { ok: false; error: string };

// Anchored to the START of the string, immediately after the scheme/user marker — this is what
// rejects an impostor host like "evilgithub.com", which a bare `/github\.com/` substring match
// would accept (cold review Finding 6): "evilgithub.com" contains "github.com" as a substring,
// so a caller could be told to fetch PR data for a repository hosted somewhere else entirely.
const GITHUB_SSH_RE = /^git@github\.com:([^/]+)\/([^/]+?)(?:\.git)?\/?$/;
const GITHUB_HTTPS_RE = /^https:\/\/github\.com\/([^/]+)\/([^/]+?)(?:\.git)?\/?$/;

// Requires the diagnostic to actually NAME the missing remote (Finding 3) — a bare exit code 2
// covers every git failure that isn't a success, including repo corruption; only a message that
// names "origin" as the missing remote is the verified quiet-empty case.
const NO_SUCH_REMOTE_RE = /no such remote ['"]?origin\b/i;

// SSH (git@github.com:owner/repo.git) and HTTPS (https://github.com/owner/repo[.git]) both
// match, and ONLY when the host is EXACTLY github.com.
export function parseGitHubRemote(url: string): { owner: string; repo: string } | null {
  const trimmed = url.trim();
  const m = GITHUB_SSH_RE.exec(trimmed) ?? GITHUB_HTTPS_RE.exec(trimmed);
  if (!m) return null;
  return { owner: m[1], repo: m[2] };
}

// Rollup mapping, §5.1's pinned table. `statusCheckRollup` is a heterogeneous list: modern
// CheckRun members carry `conclusion` (possibly null while running); legacy StatusContext
// members carry `state` and no `conclusion` at all. A member matching neither shape, or an
// unrecognised value within a shape it does match, is treated as failing — fail closed, the one
// direction that cannot make the board read greener than reality.
function classifyRollupMember(raw: unknown): "failing" | "pending" | "green" {
  if (raw === null || typeof raw !== "object") return "failing";
  const member = raw as Record<string, unknown>;
  // Branch on __typename FIRST (cold review Finding 3): a real CheckRun can legitimately omit
  // `conclusion` entirely while a check is running — spec §5.1 pins conclusion null/absent to
  // pending — but the previous `"conclusion" in member` guard treated an absent property as "not
  // this shape" and fell through to the fail-closed default, misreporting a running check as
  // failing. Falling back to the `in` checks below keeps unrecognised/legacy shapes handled the
  // same way as before.
  if (member.__typename === "CheckRun") {
    const c = member.conclusion;
    if (c === null || c === undefined) return "pending";
    if (c === "SUCCESS" || c === "NEUTRAL" || c === "SKIPPED") return "green";
    if (
      c === "FAILURE" || c === "TIMED_OUT" || c === "CANCELLED" ||
      c === "ACTION_REQUIRED" || c === "STARTUP_FAILURE" || c === "STALE"
    ) {
      return "failing";
    }
    return "failing"; // unrecognised conclusion value
  }
  if (member.__typename === "StatusContext") {
    const s = member.state;
    if (s === "SUCCESS") return "green";
    if (s === "PENDING" || s === "EXPECTED") return "pending";
    if (s === "FAILURE" || s === "ERROR") return "failing";
    return "failing"; // unrecognised state value
  }
  // An explicit __typename that is neither known shape must fail closed HERE, before the
  // property-presence fallback below — otherwise {__typename:"Unknown", conclusion:"SUCCESS"}
  // reads as green via the legacy fallback despite naming a type we don't recognise (Finding 4).
  // The fallback below stays reserved for members with NO __typename at all.
  if (member.__typename !== undefined) return "failing";
  if ("conclusion" in member) {
    const c = member.conclusion;
    if (c === null || c === undefined) return "pending";
    if (c === "SUCCESS" || c === "NEUTRAL" || c === "SKIPPED") return "green";
    if (
      c === "FAILURE" || c === "TIMED_OUT" || c === "CANCELLED" ||
      c === "ACTION_REQUIRED" || c === "STARTUP_FAILURE" || c === "STALE"
    ) {
      return "failing";
    }
    return "failing"; // unrecognised conclusion value
  }
  if ("state" in member) {
    const s = member.state;
    if (s === "SUCCESS") return "green";
    if (s === "PENDING" || s === "EXPECTED") return "pending";
    if (s === "FAILURE" || s === "ERROR") return "failing";
    return "failing"; // unrecognised state value
  }
  return "failing"; // matches neither shape
}

function rollupCiState(rollup: unknown[]): CiState {
  if (rollup.length === 0) return "none";
  let sawFailing = false;
  let sawPending = false;
  for (const member of rollup) {
    const c = classifyRollupMember(member);
    if (c === "failing") sawFailing = true;
    else if (c === "pending") sawPending = true;
  }
  if (sawFailing) return "failing";
  if (sawPending) return "pending";
  return "green";
}

type RawPr = { number: number; title: string; isDraft: boolean; url: string; statusCheckRollup: unknown[]; updatedAt: string };

// Fail closed (cold review Finding 13, then round 3's Finding 6): a row missing ANY of these
// fields, or carrying the wrong type for one, is a response shape we don't recognise, not "no CI
// configured" — gh always includes every field once it is requested, so this only fires if the
// shape changed under us. Round 2's fix validated statusCheckRollup's array-ness alone; a row like
// {"statusCheckRollup":[]} still had no number/title/isDraft/url, so Boolean(undefined) was false,
// ciState came out "none", and the row silently became {mergeReady:true} with every display/link
// field undefined — precisely the "green with no evidence" direction §5/§5.1 forbids. A genuinely
// empty rollup on an otherwise-complete row stays the valid no-CI case.
function validateRawPr(item: unknown): RawPr {
  if (item === null || typeof item !== "object") {
    throw new Error("gh pr list: PR row is not an object");
  }
  const o = item as Record<string, unknown>;
  const bad: string[] = [];
  if (typeof o.number !== "number" || !Number.isInteger(o.number)) bad.push("number");
  if (typeof o.title !== "string") bad.push("title");
  if (typeof o.isDraft !== "boolean") bad.push("isDraft");
  if (typeof o.url !== "string") bad.push("url");
  if (!Array.isArray(o.statusCheckRollup)) bad.push("statusCheckRollup");
  if (typeof o.updatedAt !== "string") bad.push("updatedAt");
  if (bad.length > 0) {
    const label = typeof o.number === "number" ? `#${o.number}` : "with no valid number";
    throw new Error(`gh pr list: PR ${label} has a missing or malformed field: ${bad.join(", ")}`);
  }
  return o as RawPr;
}

// Pure parse of `gh pr list --json number,title,isDraft,url,statusCheckRollup` output. The
// --limit is always 50 (§5.1); a response at that count cannot be distinguished from "exactly
// 50 open PRs" and is reported truncated rather than as a precise count.
export function parsePrListJson(json: string): { prs: Pr[]; truncated: boolean } {
  const raw: unknown = JSON.parse(json);
  if (!Array.isArray(raw)) throw new Error("gh pr list: expected a JSON array");
  const prs: Pr[] = raw.map((item) => {
    const r = validateRawPr(item);
    const ciState = rollupCiState(r.statusCheckRollup);
    return {
      number: r.number,
      title: r.title,
      isDraft: r.isDraft,
      url: r.url,
      ciState,
      mergeReady: !r.isDraft && (ciState === "green" || ciState === "none"),
      updatedAt: r.updatedAt,
    };
  });
  return { prs, truncated: raw.length >= 50 };
}

// Card summary (§5): worst state wins — failing beats pending beats green. "none" (no CI) only
// when no PR in the set has any check at all.
export function summarizePrState(prs: Pr[]): CiState {
  let sawPending = false;
  let sawGreen = false;
  for (const pr of prs) {
    if (pr.ciState === "failing") return "failing";
    if (pr.ciState === "pending") sawPending = true;
    else if (pr.ciState === "green") sawGreen = true;
  }
  if (sawPending) return "pending";
  if (sawGreen) return "green";
  return "none";
}

// Resolves ONE project's GitHub remote from its own repository (never the process cwd, §5.1)
// and fetches its open PRs. When the project has no git remote or the remote isn't a GitHub URL,
// returns an EXPLICIT resolved result — not null (cold review round 4, Finding 4) — since §5.1's
// "is skipped without error and shows no PR line" is a successful, known outcome: an empty PR set
// renders no PR line the same way a real zero-open-PRs remote does. This is a different outcome
// from a gh command failing (that path returns {ok:false}, still present in the result map) and
// from the project being altogether absent from the result map (reserved for "not in this
// snapshot yet" — see buildMission's join in Task 7).
export async function fetchProjectPrs(
  dir: string,
  reposRoot: string,
  run: ExecFile = systemExec,
): Promise<PrResult> {
  const repoPath = join(reposRoot, dir);
  let remoteUrl: string;
  try {
    remoteUrl = (await run("git", ["-C", repoPath, "remote", "get-url", "origin"])).trim();
  } catch (e) {
    // A verified missing-`origin` is a quiet, expected outcome (§5.1) — but git can also fail for
    // reasons that have nothing to do with the remote being absent: git itself missing (ENOENT),
    // an inaccessible path, a corrupt repo. Those are source failures and must surface as
    // {ok:false} per the error contract, not be folded into "no remote" (cold review Finding 1).
    // Exit code 2 alone is not proof — every non-"no such remote" git failure can also exit 2
    // (e.g. repo corruption), so the branch-review's Finding 3 requires the diagnostic to actually
    // name the missing remote, not just match the exit code.
    const err = e as { message?: string; code?: unknown };
    const detail = typeof err.message === "string" ? err.message : String(e);
    if (NO_SUCH_REMOTE_RE.test(detail)) {
      return { ok: true, prs: [], truncated: false };
    }
    return { ok: false, error: detail };
  }
  const parsed = parseGitHubRemote(remoteUrl);
  if (!parsed) return { ok: true, prs: [], truncated: false };
  try {
    const json = await run("gh", [
      "pr", "list",
      "-R", `${parsed.owner}/${parsed.repo}`,
      "--state", "open",
      "--limit", "50",
      "--json", "number,title,isDraft,url,statusCheckRollup,updatedAt",
    ]);
    const { prs, truncated } = parsePrListJson(json);
    return { ok: true, prs, truncated };
  } catch (e) {
    return { ok: false, error: e instanceof Error ? e.message : String(e) };
  }
}

// ponytail: a plain sequential loop — 4 carded repos today, no need for Promise.all concurrency.
// Each project's fetch is independently wrapped, so one gh failure never affects another's entry
// (§5.1 failure isolation). Every requested dir gets an entry — `fetchProjectPrs` never returns
// null (cold review round 4, Finding 4) — so a no-remote project's resolved result is never
// mistaken for one the poll hasn't reached yet.
export async function getPrsForProjects(
  dirs: string[],
  reposRoot: string,
  run: ExecFile = systemExec,
): Promise<Record<string, PrResult>> {
  // Object.create(null): `dirs` are externally sourced directory names — a plain {} would let a
  // dir literally named "__proto__" silently change the object's prototype instead of becoming an
  // enumerable entry, dropping that project from the JSON envelope (Finding 6).
  const out: Record<string, PrResult> = Object.create(null);
  for (const dir of dirs) {
    out[dir] = await fetchProjectPrs(dir, reposRoot, run);
  }
  return out;
}
