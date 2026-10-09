import { collectorResponse } from "@/server/api";
import { getRuleDocs } from "@/server/db/rules";
import { getDb } from "@/server/db";
import { getRulesData, readRuleFileRaw, rulePaths, type RawRead } from "@/server/collectors/rules";
import type { AppKey, RuleDocs } from "@/lib/rules";

export const dynamic = "force-dynamic";

// Test seam (same pattern as agent-settings): production uses these defaults,
// tests set `jaxDeps` on the Request. Keeps the disk read injectable without
// hardcoding $HOME in route tests.
export type RulesGetDeps = {
  getDocs: () => RuleDocs;
  readRuleFile: (app: AppKey) => RawRead;
};

const systemDeps: RulesGetDeps = {
  getDocs: () => getRuleDocs(getDb()),
  readRuleFile: (app) => readRuleFileRaw(rulePaths()[app]),
};

export async function GET(req: Request) {
  const deps = (req as Request & { jaxDeps?: RulesGetDeps }).jaxDeps ?? systemDeps;
  return collectorResponse(() => getRulesData(deps.getDocs(), { readRuleFile: deps.readRuleFile }));
}
