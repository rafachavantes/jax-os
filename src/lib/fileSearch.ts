import type { DirListing, SearchHit, SearchScope } from "../server/collectors/files";
import type { ContentHit } from "../server/collectors/fileContentSearch";
import { serializeScope } from "./filesWorkspace";

export type FileSearchResponse = { ok: true; data: SearchHit[]; truncated: boolean } | { ok: false; error: string };

export async function fetchFileSearch(q: string, scope: SearchScope, signal: AbortSignal): Promise<FileSearchResponse> {
  // Round-2 (d4876572f596) F4: `serializeScope` can return a repo name verbatim (`repo:<name>`) —
  // a name with `#`, `&`, `?`, or `%` would truncate or corrupt the query string unencoded. The
  // server side needs no matching decode: `new URL(req.url).searchParams.get("scope")` already
  // decodes it, same as it always has for `q`.
  const response = await fetch(`/api/files/search?q=${encodeURIComponent(q)}&scope=${encodeURIComponent(serializeScope(scope))}`, { signal });
  if (!response.ok) throw new Error("file search unavailable");
  return response.json();
}

export type ContentSearchResponse = { ok: true; data: ContentHit[]; truncated: boolean } | { ok: false; error: string };

export async function fetchContentSearch(q: string, scope: SearchScope, signal: AbortSignal): Promise<ContentSearchResponse> {
  // Round-2 (d4876572f596) F4: same reason as fetchFileSearch above.
  const response = await fetch(`/api/files/content-search?q=${encodeURIComponent(q)}&scope=${encodeURIComponent(serializeScope(scope))}`, { signal });
  if (!response.ok) throw new Error("content search unavailable");
  return response.json();
}

// §9 round-4 F1: reuses the EXISTING tree listing route, no new endpoint.
export type RepoNamesResponse = { ok: true; data: DirListing } | { ok: false; error: string };
export async function fetchRepoNames(fetchImpl: typeof fetch = fetch): Promise<RepoNamesResponse> {
  const response = await fetchImpl("/api/files/tree?root=repos&rel=");
  return response.json();
}
export function repoNamesFromListing(listing: DirListing): string[] {
  return listing.dirs.map((d) => d.name);
}

export type ContentHitGroup = { root: ContentHit["root"]; rel: string; hits: ContentHit[] };

// Review F2: content results group by file (one heading per file, each match its own row).
// Order preserved = first appearance, same file never split into two groups.
export function groupContentHits(hits: ContentHit[]): ContentHitGroup[] {
  const groups: ContentHitGroup[] = [];
  const byKey = new Map<string, ContentHitGroup>();
  for (const hit of hits) {
    const key = `${hit.root}:${hit.rel}`;
    let group = byKey.get(key);
    if (!group) {
      group = { root: hit.root, rel: hit.rel, hits: [] };
      byKey.set(key, group);
      groups.push(group);
    }
    group.hits.push(hit);
  }
  return groups;
}
