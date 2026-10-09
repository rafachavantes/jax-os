export type Envelope<T> = { ok: true; data: T } | { ok: false; error: string };

export async function fetchEnvelope<T>(url: string, signal?: AbortSignal): Promise<Extract<Envelope<T>, { ok: true }>> {
  const res = await fetch(url, { signal });
  if (!res.ok) throw new Error("unavailable");
  let json: unknown;
  try {
    json = await res.json();
  } catch (e) {
    if (e instanceof DOMException && e.name === "AbortError") throw e;
    if (e instanceof Error && e.name === "AbortError") throw e;
    throw new Error("unavailable");
  }
  if (!json || typeof json !== "object" || !("ok" in json)) throw new Error("unavailable");
  const env = json as Envelope<T>;
  if (!env.ok) throw new Error(env.error || "unavailable");
  return env;
}

export function retainOkMeta<T>(
  previous: { data: T; updatedAt: number } | undefined,
  incoming: { ok: true; data: T } | { ok: false; error: string },
  now: number,
): { data: T | undefined; error: string | undefined; updatedAt: number | undefined } {
  if (incoming.ok) return { data: incoming.data, error: undefined, updatedAt: now };
  return { data: previous?.data, error: incoming.error, updatedAt: previous?.updatedAt };
}

export type TtydConfig = {
  configured: boolean;
  reachable: boolean;
  roUrl?: string;
  rwUrl?: string;
};
