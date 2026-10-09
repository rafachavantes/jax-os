import { NextResponse } from "next/server";
import {
  composeCompare,
  productionCompareComposition,
  type CompareComposition,
} from "../../../../server/collectors/openrouter";

const CONNECTION_RE = /^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$/;

export const dynamic = "force-dynamic";

export type EndpointsRouteDeps = {
  compare: (connection: string, model: string, refresh: boolean) => Promise<unknown>;
};

const composition: CompareComposition = productionCompareComposition();

async function productionCompare(connection: string, model: string, refresh: boolean): Promise<unknown> {
  return composeCompare(composition, connection, model, refresh);
}

const systemDeps: EndpointsRouteDeps = { compare: productionCompare };

function reply(body: unknown) {
  const res = NextResponse.json(body);
  res.headers.set("Cache-Control", "no-store");
  return res;
}

function isObj(value: unknown): value is Record<string, unknown> {
  return !!value && typeof value === "object" && !Array.isArray(value);
}

async function handleGet(req: Request, deps: EndpointsRouteDeps = systemDeps) {
  const params = new URL(req.url).searchParams;
  const connection = params.get("connection");
  const model = params.get("model");
  if (!connection || !CONNECTION_RE.test(connection) || !model) {
    return reply({ ok: false, error: "invalid payload" });
  }
  try {
    const data = await deps.compare(connection, model, params.get("refresh") === "1");
    return reply({ ok: true, data });
  } catch {
    return reply({ ok: false, error: "unavailable" });
  }
}

export function GET(req: Request) {
  const injected = (req as Request & { jaxDeps?: unknown }).jaxDeps;
  return handleGet(req, isObj(injected) ? (injected as unknown as EndpointsRouteDeps) : undefined);
}
