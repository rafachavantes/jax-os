export const STALE_AFTER_SECONDS = 300;

export function isCollectionStale(
  source: { state: string; lastRefresh: number | null },
  nowSeconds: number,
): boolean {
  if (source.state === "disabled" || source.lastRefresh === null) return false;
  return nowSeconds - source.lastRefresh > STALE_AFTER_SECONDS;
}

export function isUnresolvedModel(name: string): boolean {
  return /\/jaxflow-builder-(default|fallback)$/.test(name);
}
