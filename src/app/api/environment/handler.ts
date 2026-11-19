import { NextResponse } from "next/server";
import { collectorResponse } from "@/server/api";
import {
  createEnvironmentRuntime, systemEnvironmentDeps, type EnvironmentRuntime,
} from "@/server/collectors/environment";

export const ENVIRONMENT_SECTIONS = ["components", "resources", "workloads", "packages"] as const;
export type EnvironmentSection = (typeof ENVIRONMENT_SECTIONS)[number];

// Fixed enum only: absent defaults to components, any other supplied value is refused.
export function parseSection(raw: string | null): EnvironmentSection | null {
  if (raw === null || raw === "") return "components";
  return (ENVIRONMENT_SECTIONS as readonly string[]).includes(raw) ? (raw as EnvironmentSection) : null;
}

export type EnvironmentHandlerDeps = { runtime: EnvironmentRuntime };

let sharedRuntime: EnvironmentRuntime | null = null;

function systemDeps(): EnvironmentHandlerDeps {
  sharedRuntime ??= createEnvironmentRuntime(systemEnvironmentDeps());
  return { runtime: sharedRuntime };
}

// Thin read-only section route: each call touches exactly one section, so a slow package probe
// can never delay components or resources (they are independent polls on the client).
export async function handleEnvironmentGet(req: Request, deps: EnvironmentHandlerDeps = systemDeps()) {
  const section = parseSection(new URL(req.url).searchParams.get("section"));
  if (section === null) {
    return NextResponse.json({ ok: false, error: "invalid section" });
  }
  return collectorResponse<unknown>(() => deps.runtime[section]());
}
