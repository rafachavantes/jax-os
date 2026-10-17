// Inventory collector (spec §7). ALL native access goes through the finite
// Python helper `scripts/jaxflow_inventory_io.py`; this module only derives
// the fixed managed roots, invokes the helper with a bounded timeout, and
// parses its JSON envelope. Read-only snapshot uses the system interpreter;
// Codex writes use the fixed isolated venv (clients cannot choose an
// executable). Nothing here accepts a client-supplied path.

import { execFile } from "node:child_process";
import { existsSync } from "node:fs";
import { homedir } from "node:os";
import { join } from "node:path";
import { jaxosHome } from "../env";

export type InventoryExecutor = "claude" | "codex" | "opencode";
export type InventoryKind = "plugin" | "skill" | "agent";
export type InventoryChange = "disable" | "enable" | "remove-registration" | "remove-installation";

export type InventoryCapability = { available: boolean; reason: string | null };
export type InventoryItemOperation = {
  operationId: string;
  change: string;
  stage: string;
  effect: string | null;
  /** Server-derived source owner, so absence is proven from its healthy bucket. */
  executor?: string;
  /** The op's own source bucket could not be read: unavailable, not removed. */
  unavailable?: boolean;
};
export type InventoryRecovery = InventoryItemOperation & { itemId: string };
export type InventoryItem = {
  id: string;
  executor: InventoryExecutor;
  kind: InventoryKind;
  registration: string;
  name: string;
  description: string;
  path: string;
  scope: string;
  origin: string;
  parent: string | null;
  canonicalTarget: string | null;
  state: "enabled" | "disabled" | "restricted" | "parent-disabled" | "unknown" | "broken";
  capabilities: Partial<Record<InventoryChange, InventoryCapability>>;
  /** Claude plugin cache facts, folded onto the configured logical plugin. */
  installedVersion?: string;
  cachePath?: string;
  /** Served reload recovery: the unresolved operation bound to this item. */
  operation?: InventoryItemOperation;
};

export type InventoryExecutorBucket = { ok: true; items: InventoryItem[] } | { ok: false; error: string; items: [] };
export type InventoryData = {
  executors: Record<InventoryExecutor, InventoryExecutorBucket>;
  /** Operations whose item is no longer discovered, kept as recovery rows. */
  recovery?: InventoryRecovery[];
};

export type InventoryPreview = {
  itemId: string;
  change: InventoryChange;
  revision: string;
  targets: string[];
  effect: string;
  recovery: string | null;
  requiresNewSession: boolean;
};

export type InventoryRevisionPair = { settings: string; sources: Record<string, string> };

// Readback of an item's actual native state (plus the OpenCode settings revision
// when the recorded executor is known). `absent` is authoritative ONLY for the
// item's own healthy source bucket.
export type InventoryReadback = { revision: string; settings: InventoryRevisionPair | null };

// Protected preparation (never sent to the browser): the checked before/candidate
// digests and the bounded restoration record, computed before any native write.
export type InventoryPreparation = {
  itemId: string;
  change: InventoryChange;
  beforeDigest: string;
  candidateDigest: string;
  provenance: unknown;
  settingsExpected: InventoryRevisionPair | null;
  settingsCandidate: InventoryRevisionPair | null;
  executor: InventoryExecutor;
  selector: string;
  target: string;
};

export type HelperRun = (
  action: "snapshot" | "preview" | "prepare" | "revision" | "apply" | "reconcile",
  payload: Record<string, unknown>,
  python: string,
) => Promise<{ ok: true; data: unknown } | { ok: false; error: string }>;

export type InventoryDeps = {
  run: HelperRun;
  roots: Record<string, string>;
  /** Narrows Codex-write capability: absent ⇒ the whole mutating surface is setup-required. */
  venvPython: string | null;
};

const HELPER = join(process.cwd(), "scripts", "jaxflow_inventory_io.py");
export const HELPER_TIMEOUT_MS = 10_000;

function defaultRoots(): Record<string, string> {
  const home = homedir();
  return {
    claude_dir: join(home, ".claude"),
    codex_dir: join(home, ".codex"),
    opencode_dir: join(home, ".config", "opencode"),
    jax_os: jaxosHome(),
  };
}

// The fixed production write interpreter (never client-selected).
export function defaultVenvPython(): string | null {
  const path = join(jaxosHome(), "inventory-venv", "bin", "python");
  return existsSync(path) ? path : null;
}

export function runPython(python: string, action: string, payload: unknown): Promise<{ ok: true; data: unknown } | { ok: false; error: string }> {
  return new Promise((resolve) => {
    let settled = false;
    const child = execFile(
      python,
      [HELPER, action],
      { timeout: HELPER_TIMEOUT_MS, maxBuffer: 4 * 1024 * 1024 },
      (err, stdout) => {
        if (settled) return;
        settled = true;
        if (err) {
          // A timeout is UNCONFIRMED, never automatically "not applied".
          const killed = (err as { killed?: boolean }).killed === true;
          resolve({ ok: false, error: killed ? "inventory-unconfirmed" : "inventory-helper-failed" });
          return;
        }
        try {
          const parsed = JSON.parse(stdout) as { ok?: unknown; data?: unknown; error?: unknown };
          if (parsed.ok === true) resolve({ ok: true, data: parsed.data });
          else resolve({ ok: false, error: typeof parsed.error === "string" ? parsed.error : "inventory-helper-failed" });
        } catch {
          resolve({ ok: false, error: "inventory-helper-failed" });
        }
      },
    );
    child.stdin?.end(JSON.stringify(payload));
  });
}

export function systemInventoryDeps(): InventoryDeps {
  return {
    run: (action, payload, python) => runPython(python, action, payload),
    roots: defaultRoots(),
    venvPython: defaultVenvPython(),
  };
}

function requireVenv(deps: InventoryDeps): string {
  if (!deps.venvPython) throw new Error("inventory-setup-required");
  return deps.venvPython;
}

export async function getInventoryData(deps: InventoryDeps = systemInventoryDeps()): Promise<InventoryData> {
  const result = await deps.run("snapshot", { roots: deps.roots }, "python3");
  if (!result.ok) throw new Error(result.error);
  return result.data as InventoryData;
}

export async function previewInventory(
  deps: InventoryDeps,
  itemId: string,
  change: InventoryChange,
  provenance?: unknown,
): Promise<InventoryPreview> {
  const result = await deps.run("preview", { roots: deps.roots, itemId, change, provenance }, "python3");
  if (!result.ok) throw new Error(result.error);
  return result.data as InventoryPreview;
}

// Internal read-only preparation (server-side only). It performs no native
// write, but a Codex candidate needs the pinned tomlkit, so it runs on the
// fixed isolated interpreter like apply.
export async function prepareInventory(
  deps: InventoryDeps,
  itemId: string,
  change: InventoryChange,
  expectedRevision: string,
  provenance?: unknown,
  operationId?: string,
): Promise<InventoryPreparation> {
  const python = requireVenv(deps);
  const result = await deps.run("prepare", { roots: deps.roots, itemId, change, expectedRevision, provenance, operationId }, python);
  if (!result.ok) throw new Error(result.error);
  return result.data as InventoryPreparation;
}

// Read-only current revision for recheck: never preview eligibility, so an
// already-settled action still reads back its true native state ("absent" for a
// genuine removal). The recorded executor narrows the read to the op's own
// source so a failed unrelated bucket can never fabricate absence.
export async function inventoryRevision(deps: InventoryDeps, itemId: string, executor?: string): Promise<InventoryReadback> {
  const result = await deps.run("revision", { roots: deps.roots, itemId, executor }, "python3");
  if (!result.ok) throw new Error(result.error);
  return result.data as InventoryReadback;
}

export async function applyInventory(
  deps: InventoryDeps,
  itemId: string,
  change: InventoryChange,
  expectedRevision: string,
  operationId: string,
  provenance?: unknown,
): Promise<unknown> {
  // Only Codex needs the pinned tomlkit; the helper imports it lazily, but the
  // write path always runs on the fixed venv so the import is available.
  const python = requireVenv(deps);
  const result = await deps.run("apply", { roots: deps.roots, itemId, change, expectedRevision, operationId, provenance }, python);
  if (!result.ok) throw new Error(result.error);
  return result.data;
}

export async function reconcileInventory(
  deps: InventoryDeps,
  expected: unknown,
  operationId: string,
): Promise<unknown> {
  const python = requireVenv(deps);
  const result = await deps.run("reconcile", { roots: deps.roots, expected, operationId }, python);
  if (!result.ok) throw new Error(result.error);
  return result.data;
}
