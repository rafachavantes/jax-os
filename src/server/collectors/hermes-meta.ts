import { execFile } from "node:child_process";
import { readFileSync } from "node:fs";
import { homedir } from "node:os";
import { join } from "node:path";
import { promisify } from "node:util";

const pExecFile = promisify(execFile);

type Json = Record<string, unknown>;
const num = (v: unknown): number | null =>
  typeof v === "number" && Number.isFinite(v) ? v : null;

export type PromptComponent = {
  key: "systemPrompt" | "skillsIndex" | "memory" | "userProfile" | "toolSchemas";
  bytes: number;
};

export type PromptBudget = {
  platform: string | null;
  model: string | null;
  components: PromptComponent[];
  totalBytes: number;
  toolCount: number;
};

const PROMPT_KEYS = ["system_prompt", "skills_index", "tools", "platform", "model"];

export function parsePromptSize(stdout: string): PromptBudget {
  // Banner noise may itself contain braces (even parseable JSON fragments):
  // scan each `{` until one parses AND looks like the prompt-size payload.
  let o: Json | null = null;
  for (let i = stdout.indexOf("{"); i !== -1; i = stdout.indexOf("{", i + 1)) {
    try {
      const parsed = JSON.parse(stdout.slice(i)) as Json;
      if (PROMPT_KEYS.some((k) => k in parsed)) {
        o = parsed;
        break;
      }
    } catch {
      /* keep scanning */
    }
  }
  if (!o) throw new Error("no json in prompt-size output");
  const bytes = (v: unknown) => num((v as Json | undefined)?.bytes) ?? 0;
  const tools = (o.tools ?? {}) as Json;
  const components: PromptComponent[] = [
    { key: "systemPrompt", bytes: bytes(o.system_prompt) },
    { key: "skillsIndex", bytes: bytes(o.skills_index) },
    { key: "memory", bytes: bytes(o.memory) },
    { key: "userProfile", bytes: bytes(o.user_profile) },
    { key: "toolSchemas", bytes: num(tools.json_bytes) ?? 0 },
  ];
  return {
    platform: typeof o.platform === "string" ? o.platform : null,
    model: typeof o.model === "string" ? o.model : null,
    components,
    totalBytes: components.reduce((s, c) => s + c.bytes, 0),
    toolCount: num(tools.count) ?? 0,
  };
}

export type SkillActivity = {
  name: string;
  useCount: number;
  lastUsedAt: string | null;
  flagged: boolean;
};

export function parseSkillsUsage(jsonText: string, nowMs: number): SkillActivity[] {
  const o = JSON.parse(jsonText) as Record<string, Json>;
  const THIRTY_D = 30 * 86_400_000;
  return Object.entries(o)
    .map(([name, s]) => {
      const useCount = num(s.use_count) ?? 0;
      const lastUsedAt = typeof s.last_used_at === "string" ? s.last_used_at : null;
      // Unparseable timestamp fails safe toward stale (NaN comparisons are false).
      const t = lastUsedAt ? Date.parse(lastUsedAt) : NaN;
      const stale = Number.isNaN(t) || nowMs - t > THIRTY_D;
      return {
        name,
        useCount,
        lastUsedAt,
        flagged: s.state === "active" && (useCount === 0 || stale),
      };
    })
    .sort((a, b) => b.useCount - a.useCount);
}

// ---- glue (untested) ----

export type SectionResult<T> = { ok: true; data: T } | { ok: false; error: string };
export type HermesMeta = {
  prompt: SectionResult<PromptBudget>;
  skills: SectionResult<SkillActivity[]>;
};

const HERMES_BIN = join(homedir(), ".local", "bin", "hermes");
const SKILLS_USAGE = join(homedir(), ".hermes", "skills", ".usage.json");

// ponytail: cache both success AND failure for 10min — the CLI takes seconds
// to boot (venv + secrets); hammering a broken one every poll helps nobody.
// ponytail: no in-flight lock — concurrent cache-miss polls may double-spawn
// the CLI; add a pending-promise guard if that ever matters.
let promptCache: { at: number; res: SectionResult<PromptBudget> } | null = null;

const HERMES_OPTS = { encoding: "utf8" as const, timeout: 30_000, maxBuffer: 2 * 1024 * 1024 };

export type PromptExec = (
  file: string,
  args: readonly string[],
  options: { encoding: "utf8"; timeout: number; maxBuffer: number },
) => Promise<{ stdout: string }>;

export function resetHermesPromptCache(): void {
  promptCache = null;
}

function classifyPromptExecError(e: unknown): "timeout" | "unavailable" | "invalid-output" {
  const err = e as { code?: unknown; killed?: unknown };
  if (err.code === "ETIMEDOUT" || err.killed === true) return "timeout";
  if (err.code === "ERR_CHILD_PROCESS_STDIO_MAXBUFFER") return "invalid-output";
  return "unavailable";
}

export async function getHermesMeta(exec: PromptExec = pExecFile as PromptExec): Promise<HermesMeta> {
  let prompt: SectionResult<PromptBudget>;
  if (promptCache && Date.now() - promptCache.at < 600_000) {
    prompt = promptCache.res;
  } else {
    try {
      const { stdout } = await exec(HERMES_BIN, ["prompt-size", "--json"], HERMES_OPTS);
      try {
        prompt = { ok: true, data: parsePromptSize(stdout) };
      } catch {
        prompt = { ok: false, error: "invalid-output" };
      }
    } catch (e) {
      prompt = { ok: false, error: classifyPromptExecError(e) };
    }
    promptCache = { at: Date.now(), res: prompt };
  }

  let skills: SectionResult<SkillActivity[]>;
  try {
    skills = { ok: true, data: parseSkillsUsage(readFileSync(SKILLS_USAGE, "utf8"), Date.now()) };
  } catch (e) {
    skills = { ok: false, error: e instanceof Error ? e.message : String(e) };
  }

  return { prompt, skills };
}
