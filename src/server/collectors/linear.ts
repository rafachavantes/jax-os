import { MutationRejected } from "../../lib/mutationOutcome";

export type LinearTeam = { id: string; key: string; name: string };
export type KanbanState = {
  id: string;
  name: string;
  color: string;
  position: number;
  type: string;
};
export type KanbanLabel = { id: string; name: string; color: string };
export type LinearMember = { id: string; displayName: string };
export type KanbanIssue = {
  id: string;
  identifier: string;
  title: string;
  priority: number;
  url: string;
  stateId: string;
  createdAt: string;
  updatedAt: string;
  project?: string;
  projectId?: string;
  assignee?: string;
  assigneeId?: string;
  labels: KanbanLabel[];
  subIssue?: { total: number; done: number }; // present only when the issue has children
};
export type KanbanBoard = { states: KanbanState[]; issues: KanbanIssue[] };

export type SubIssue = { identifier: string; title: string; stateName: string; stateType: string; assignee?: string };
export type IssueComment = { id: string; author: string; body: string; createdAt: string; pending?: boolean };
export type IssueAttachment = { id: string; title: string; url: string };
export type IssueParentLink = { id: string; identifier: string; title: string; url: string };
export type IssueDetail = {
  description: string;
  subIssue: { total: number; done: number };
  children: SubIssue[];
  comments: IssueComment[];
  attachments: IssueAttachment[];
  parent?: IssueParentLink;
  commentsTruncated: boolean;
  childrenTruncated: boolean;
  commentsIncomplete: boolean;
  childrenIncomplete: boolean;
  issue: KanbanIssue;
  teamId: string;
  states: KanbanState[];
};

// Explicit distinction from transport/GraphQL failure: the issue id is valid
// but Linear has no such issue. The route maps this to a stable not-found
// envelope ({ok:false, code:"issue-not-found"}), never to "unavailable".
export class IssueNotFound extends Error {
  constructor() { super("issue-not-found"); }
}

const EXCLUDED_TYPES = ["canceled", "duplicate", "triage"];
// Linear scopes `position` per state TYPE — ranking types first keeps
// Done after In Review (verified live: Done pos=3, In Review pos=1002)
const STATE_TYPE_RANK: Record<string, number> = {
  triage: 0,
  backlog: 1,
  unstarted: 2,
  started: 3,
  completed: 4,
};

/* eslint-disable @typescript-eslint/no-explicit-any */
export function mapTeams(json: unknown): LinearTeam[] {
  const nodes = (json as any)?.data?.teams?.nodes;
  if (!Array.isArray(nodes)) throw new Error("unexpected teams response");
  return nodes.map((n: any) => ({
    id: String(n.id),
    key: String(n.key),
    name: String(n.name),
  }));
}

// board card counter: done = completed state type. undefined (no counter) when childless.
function countSubIssues(nodes: any): { total: number; done: number } | undefined {
  if (!Array.isArray(nodes) || nodes.length === 0) return undefined;
  return { total: nodes.length, done: nodes.filter((c: any) => c.state?.type === "completed").length };
}

// single issue mapper shared by the board and the detail read — one shape for
// the same Linear node type, so a card and its detail can never disagree.
function mapIssue(n: any): KanbanIssue {
  return {
    id: String(n.id),
    identifier: String(n.identifier),
    title: String(n.title),
    priority: Number(n.priority ?? 0),
    url: String(n.url),
    stateId: String(n.state?.id ?? ""),
    createdAt: String(n.createdAt),
    updatedAt: String(n.updatedAt),
    project: n.project?.name ?? undefined,
    projectId: n.project?.id ?? undefined,
    assignee: n.assignee?.displayName ?? undefined,
    assigneeId: n.assignee?.id ?? undefined,
    labels: Array.isArray(n.labels?.nodes)
      ? n.labels.nodes.map((l: any) => ({ id: String(l.id), name: String(l.name), color: String(l.color) }))
      : [],
    subIssue: countSubIssues(n.children?.nodes),
  };
}

export function mapBoard(json: unknown): KanbanBoard {
  const team = (json as any)?.data?.team;
  const stateNodes = team?.states?.nodes;
  const issueNodes = team?.issues?.nodes;
  if (!Array.isArray(stateNodes) || !Array.isArray(issueNodes)) {
    throw new Error("unexpected board response");
  }
  const states: KanbanState[] = stateNodes
    .filter((s: any) => !EXCLUDED_TYPES.includes(s.type))
    .map((s: any) => ({
      id: String(s.id),
      name: String(s.name),
      color: String(s.color),
      position: Number(s.position),
      type: String(s.type),
    }))
    .sort(
      (a: KanbanState, b: KanbanState) =>
        (STATE_TYPE_RANK[a.type] ?? 9) - (STATE_TYPE_RANK[b.type] ?? 9) ||
        a.position - b.position,
    );
  const issues: KanbanIssue[] = issueNodes.map(mapIssue);
  return { states, issues };
}

export function mapProjectCounts(json: unknown): Record<string, number> {
  const nodes = (json as any)?.data?.issues?.nodes;
  if (!Array.isArray(nodes)) throw new Error("unexpected issues response");
  const counts: Record<string, number> = {};
  for (const n of nodes) {
    const name = (n as any)?.project?.name;
    if (typeof name === "string" && name) counts[name] = (counts[name] ?? 0) + 1;
  }
  return counts;
}

function mapSubIssue(c: any): SubIssue {
  return {
    identifier: String(c.identifier),
    title: String(c.title),
    stateName: String(c.state?.name ?? ""),
    stateType: String(c.state?.type ?? ""),
    assignee: c.assignee?.displayName ?? undefined,
  };
}

function mapComment(m: any): IssueComment {
  return {
    id: String(m.id),
    // agent/integration comments carry botActor, not user — null-guard both
    author: String(m.user?.displayName ?? m.botActor?.name ?? "—"),
    body: String(m.body ?? ""),
    createdAt: String(m.createdAt ?? ""),
  };
}

// ascending by trailing issue number (API returns descending); MOA-8 before MOA-9
function sortChildren(children: SubIssue[]): SubIssue[] {
  return [...children].sort(
    (a, b) => (Number(a.identifier.split("-").pop()) || 0) - (Number(b.identifier.split("-").pop()) || 0),
  );
}

// Pulls a {nodes, pageInfo} connection off a raw GraphQL node, mapping each
// entry with mapFn. A missing/malformed connection degrades to empty and not
// truncated rather than throwing — the detail's core fields are already
// validated by the time this runs; comments/children/pageInfo are supplementary.
function extractConnection<T>(conn: any, mapFn: (n: any) => T): Connection<T> {
  const nodes = Array.isArray(conn?.nodes) ? conn.nodes.map(mapFn) : [];
  const pageInfo = conn?.pageInfo;
  return {
    nodes,
    pageInfo: {
      hasNextPage: pageInfo?.hasNextPage === true,
      endCursor: typeof pageInfo?.endCursor === "string" ? pageInfo.endCursor : null,
    },
  };
}

export function mapIssueDetail(json: unknown): IssueDetail {
  const issue = (json as any)?.data?.issue;
  if (issue === null) throw new IssueNotFound();
  if (!issue) throw new Error("unexpected issue response");
  const stateNodes = issue?.team?.states?.nodes;
  // identity/title/team/state are the detail contract's skeleton: assert before
  // mapping so malformed input can never surface as a String(undefined) card.
  if (
    typeof issue?.team?.id !== "string" || !issue.team.id ||
    typeof issue?.id !== "string" || !issue.id ||
    typeof issue?.identifier !== "string" || !issue.identifier ||
    typeof issue?.title !== "string" || !issue.title ||
    typeof issue?.state?.id !== "string" || !issue.state.id ||
    !Array.isArray(stateNodes)
  ) {
    throw new Error("unexpected issue response");
  }
  const children = sortChildren(extractConnection(issue.children, mapSubIssue).nodes);
  const comments = extractConnection(issue.comments, mapComment).nodes;
  const attachments: IssueAttachment[] = Array.isArray(issue.attachments?.nodes)
    ? issue.attachments.nodes.map((a: any) => ({ id: String(a.id), title: String(a.title), url: String(a.url) }))
    : [];
  const parent: IssueParentLink | undefined = issue.parent
    ? { id: String(issue.parent.id), identifier: String(issue.parent.identifier), title: String(issue.parent.title), url: String(issue.parent.url) }
    : undefined;
  // detail states keep EVERY team state (canceled/triage included): an off-board
  // issue may legitimately sit on one. Same rank ordering as the board; the
  // board's excluded-type filtering stays board-only.
  const states: KanbanState[] = stateNodes
    .map((s: any) => ({
      id: String(s.id),
      name: String(s.name),
      color: String(s.color),
      position: Number(s.position),
      type: String(s.type),
    }))
    .sort(
      (a: KanbanState, b: KanbanState) =>
        (STATE_TYPE_RANK[a.type] ?? 9) - (STATE_TYPE_RANK[b.type] ?? 9) ||
        a.position - b.position,
    );
  return {
    description: String(issue.description ?? ""),
    subIssue: { total: children.length, done: children.filter((c) => c.stateType === "completed").length },
    children,
    comments,
    attachments,
    parent,
    // page 1 only here — getIssueDetail overwrites all four once its pagination
    // loop resolves; a direct mapIssueDetail() call (as in the fixtures above)
    // has no later page to check, so "not truncated" is the honest default.
    commentsTruncated: false,
    childrenTruncated: false,
    commentsIncomplete: false,
    childrenIncomplete: false,
    issue: mapIssue(issue),
    teamId: String(issue.team.id),
    states,
  };
}

export function mapMembers(json: unknown): LinearMember[] {
  const nodes = (json as any)?.data?.team?.members?.nodes;
  if (!Array.isArray(nodes)) throw new Error("unexpected members response");
  return nodes.map((n: any) => ({ id: String(n.id), displayName: String(n.displayName) }));
}

export function mapLabels(json: unknown): KanbanLabel[] {
  const nodes = (json as any)?.data?.team?.labels?.nodes;
  if (!Array.isArray(nodes)) throw new Error("unexpected labels response");
  return nodes.map((n: any) => ({ id: String(n.id), name: String(n.name), color: String(n.color) }));
}
/* eslint-enable @typescript-eslint/no-explicit-any */

const LINEAR_URL = "https://api.linear.app/graphql";

// SP0 read cache: kanban polls (10s) re-fire every GraphQL read with zero
// caching on the 2500/h bucket SHARED WITH HERMES. 30s TTL cuts most of it.
// Opt-in per call: mutations must NEVER cache (a repeat mutation within the
// TTL would be silently swallowed) and every successful mutation clears the
// cache so the next board read reflects the write.
const CACHE_TTL_MS = 30_000;
type CacheEntry = { at: number; json: unknown };
const responseCache = new Map<string, CacheEntry>();

export function cacheGet(cache: Map<string, CacheEntry>, key: string, now: number, ttl: number = CACHE_TTL_MS): unknown | null {
  const hit = cache.get(key);
  return hit && now - hit.at < ttl ? hit.json : null;
}

export function cachePut(cache: Map<string, CacheEntry>, key: string, json: unknown, now: number): void {
  cache.set(key, { at: now, json });
}

const TEAMS_QUERY = `{ teams { nodes { id key name } } }`;

// ponytail: first 250 covers MOA's 113 issues today with headroom —
// paginate when a team outgrows 250. children(first:50) counts sub-issues per
// card; a parent with >50 sub-issues would undercount (phases have ≤13) — raise then.
const BOARD_QUERY = `query Board($teamId: String!) {
  team(id: $teamId) {
    states { nodes { id name color position type } }
    issues(
      first: 250
      orderBy: updatedAt
      filter: {
        parent: { null: true }
        state: { type: { nin: ["canceled", "duplicate", "triage"] } }
        or: [{ completedAt: { null: true } }, { completedAt: { gt: "-P1W" } }]
      }
    ) {
      nodes {
        id identifier title priority url
        createdAt updatedAt
        state { id }
        project { id name }
        assignee { id displayName }
        labels { nodes { id name color } }
        children(first: 50) { nodes { state { type } } }
      }
    }
  }
}`;

// Generic result shape for any Linear Relay-style connection (comments,
// children, attachments, …): { nodes, pageInfo { hasNextPage endCursor } }.
// Exported (F2) — Part 2 imports this type by name.
export type Connection<T> = { nodes: T[]; pageInfo: { hasNextPage: boolean; endCursor: string | null } };

// jax-os UI ceiling, not a Linear limit: past 1,000 items we stop and flag
// truncated rather than fetch forever (phase 6 spec §8.3). Exported (F2) —
// Part 2 imports this constant by name.
export const PAGE_CEILING = 1000;

// Generic cursor-pagination loop shared by comments and children: starts from
// an already-fetched first page, keeps calling fetchPage(cursor) while more
// remain, stops at PAGE_CEILING items even if Linear says there's more.
// F1: a page claiming hasNextPage with no cursor, a repeated cursor, or zero
// nodes can never progress — abort with truncated:true instead of spinning
// forever. Round-2 F3: fetchPage/extract run inside a try/catch — a rejected fetch or
// a throwing extract (malformed page) is caught the same way, so this never
// throws and a first page already loaded always stays renderable.
// Round-3 F1: `reason` tells a genuine ceiling apart from every other early stop, since the UI reads "truncated" as "first 1,000 shown".
export async function fetchAllPaginated<T>(
  firstPage: Connection<T>,
  fetchPage: (cursor: string) => Promise<unknown>,
  extract: (json: unknown) => Connection<T>,
): Promise<{ nodes: T[]; truncated: boolean; reason: "ceiling" | "incomplete" | null }> {
  let nodes = firstPage.nodes;
  let pageInfo = firstPage.pageInfo;
  const seenCursors = new Set<string>();
  while (pageInfo.hasNextPage && nodes.length < PAGE_CEILING) {
    const cursor = pageInfo.endCursor;
    if (!cursor || seenCursors.has(cursor)) {
      return { nodes: nodes.slice(0, PAGE_CEILING), truncated: true, reason: "incomplete" };
    }
    seenCursors.add(cursor);
    let next: Connection<T>;
    try {
      const json = await fetchPage(cursor);
      next = extract(json);
    } catch {
      return { nodes: nodes.slice(0, PAGE_CEILING), truncated: true, reason: "incomplete" };
    }
    if (next.nodes.length === 0 && next.pageInfo.hasNextPage) {
      return { nodes: nodes.slice(0, PAGE_CEILING), truncated: true, reason: "incomplete" };
    }
    nodes = nodes.concat(next.nodes);
    pageInfo = next.pageInfo;
  }
  // Diff review a7183f365215 F1: hasNextPage alone misses a final page that
  // crosses PAGE_CEILING while itself reporting no further pages — check the
  // accumulated count too, so the slice is never silently unflagged.
  const ceiling = pageInfo.hasNextPage || nodes.length > PAGE_CEILING;
  return { nodes: nodes.slice(0, PAGE_CEILING), truncated: ceiling, reason: ceiling ? "ceiling" : null };
}

const COUNTS_QUERY = `{ issues(first: 250, filter: { state: { type: { in: ["backlog", "unstarted", "started"] } } }) { nodes { project { name } } } }`;

const UPDATE_MUTATION = `mutation Update($id: String!, $input: IssueUpdateInput!) {
  issueUpdate(id: $id, input: $input) { success }
}`;

const COMMENT_MUTATION = `mutation Comment($issueId: String!, $body: String!) {
  commentCreate(input: { issueId: $issueId, body: $body }) { success }
}`;

const ISSUE_DETAIL_QUERY = `query IssueDetail($id: String!, $commentsCursor: String, $childrenCursor: String) {
  issue(id: $id) {
    id identifier title priority url
    createdAt updatedAt
    state { id }
    project { id name }
    assignee { id displayName }
    labels { nodes { id name color } }
    team { id states { nodes { id name color position type } } }
    description
    attachments { nodes { id title url } }
    parent { id identifier title url }
    children(first: 100, after: $childrenCursor) {
      nodes { identifier title state { name type } assignee { displayName } }
      pageInfo { hasNextPage endCursor }
    }
    comments(first: 100, after: $commentsCursor) {
      nodes { id body createdAt user { displayName } botActor { name } }
      pageInfo { hasNextPage endCursor }
    }
  }
}`;

const MEMBERS_QUERY = `query Members($teamId: String!) { team(id: $teamId) { members { nodes { id displayName } } } }`;

const LABELS_QUERY = `query Labels($teamId: String!) { team(id: $teamId) { labels { nodes { id name color } } } }`;

async function linearFetch(
  query: string,
  variables?: Record<string, unknown>,
  useCache = false,
): Promise<unknown> {
  const cacheKey = useCache ? query + JSON.stringify(variables ?? {}) : null;
  if (cacheKey) {
    const hit = cacheGet(responseCache, cacheKey, Date.now());
    if (hit) return hit;
  }
  const key = process.env.LINEAR_API_KEY;
  if (!key) throw new MutationRejected("LINEAR_API_KEY not configured");
  const res = await fetch(LINEAR_URL, {
    method: "POST",
    // personal API keys go RAW in Authorization — Linear rejects "Bearer <key>"
    headers: { "Content-Type": "application/json", Authorization: key },
    body: JSON.stringify({ query, variables }),
    signal: AbortSignal.timeout(5000),
  });
  if (!res.ok) throw new Error(`linear responded ${res.status}`);
  const json = (await res.json()) as { errors?: { message?: string }[] } | null;
  if (json?.errors?.length) throw new Error(json.errors[0]?.message ?? "linear graphql error");
  if (cacheKey) cachePut(responseCache, cacheKey, json, Date.now());
  return json;
}

export async function getTeams(): Promise<LinearTeam[]> {
  return mapTeams(await linearFetch(TEAMS_QUERY, undefined, true));
}

export async function getBoard(teamId: string): Promise<KanbanBoard> {
  return mapBoard(await linearFetch(BOARD_QUERY, { teamId }, true));
}

export async function getProjectCounts(): Promise<Record<string, number>> {
  return mapProjectCounts(await linearFetch(COUNTS_QUERY, undefined, true));
}

export type IssuePatch = { stateId?: string; priority?: number; assigneeId?: string | null; labelIds?: string[]; title?: string; description?: string };
export type IssueCreateInput = {
  teamId: string;
  title: string;
  description?: string;
  projectId?: string | null;
  stateId?: string;
  priority?: number;
  assigneeId?: string | null;
  labelIds?: string[];
};

export async function updateIssue(id: string, input: IssuePatch): Promise<void> {
  const json = await linearFetch(UPDATE_MUTATION, { id, input }) as { data?: { issueUpdate?: { success?: boolean } } } | null;
  const success = json?.data?.issueUpdate?.success;
  if (success === false) throw new MutationRejected("issueUpdate did not succeed");
  if (success !== true) throw new Error("issueUpdate outcome unconfirmed");
  responseCache.clear();
}

export async function moveIssue(issueId: string, stateId: string): Promise<void> {
  return updateIssue(issueId, { stateId });
}

export async function createComment(issueId: string, body: string): Promise<void> {
  const json = await linearFetch(COMMENT_MUTATION, { issueId, body }) as { data?: { commentCreate?: { success?: boolean } } } | null;
  const success = json?.data?.commentCreate?.success;
  if (success === false) throw new MutationRejected("commentCreate did not succeed");
  if (success !== true) throw new Error("commentCreate outcome unconfirmed");
  responseCache.clear();
}

const ISSUE_CREATE_MUTATION = `mutation Create($input: IssueCreateInput!) {
  issueCreate(input: $input) { success issue { id identifier } }
}`;

export async function createIssue(input: IssueCreateInput): Promise<{ id: string; identifier: string }> {
  const json = await linearFetch(ISSUE_CREATE_MUTATION, { input }) as
    { data?: { issueCreate?: { success?: boolean; issue?: { id?: string; identifier?: string } } } } | null;
  const payload = json?.data?.issueCreate;
  if (payload?.success === false) throw new MutationRejected("issueCreate did not succeed");
  if (payload?.success !== true || typeof payload.issue?.id !== "string" || typeof payload.issue?.identifier !== "string") {
    throw new Error("issueCreate outcome unconfirmed");
  }
  responseCache.clear();
  return { id: payload.issue.id, identifier: payload.issue.identifier };
}

export async function getIssueDetail(id: string): Promise<IssueDetail> {
  const json = await linearFetch(ISSUE_DETAIL_QUERY, { id, commentsCursor: null, childrenCursor: null }, true);
  const detail = mapIssueDetail(json);
  const issueNode = (json as any)?.data?.issue;

  const commentsFirst = extractConnection(issueNode?.comments, mapComment);
  const commentsResult = await fetchAllPaginated(
    commentsFirst,
    (cursor) => linearFetch(ISSUE_DETAIL_QUERY, { id, commentsCursor: cursor, childrenCursor: null }, false),
    (j) => extractConnection((j as any)?.data?.issue?.comments, mapComment),
  );
  detail.comments = commentsResult.nodes;
  detail.commentsTruncated = commentsResult.reason === "ceiling";
  detail.commentsIncomplete = commentsResult.reason === "incomplete";

  const childrenFirst = extractConnection(issueNode?.children, mapSubIssue);
  const childrenResult = await fetchAllPaginated(
    childrenFirst,
    (cursor) => linearFetch(ISSUE_DETAIL_QUERY, { id, commentsCursor: null, childrenCursor: cursor }, false),
    (j) => extractConnection((j as any)?.data?.issue?.children, mapSubIssue),
  );
  const children = sortChildren(childrenResult.nodes);
  detail.children = children;
  detail.childrenTruncated = childrenResult.reason === "ceiling";
  detail.childrenIncomplete = childrenResult.reason === "incomplete";
  detail.subIssue = { total: children.length, done: children.filter((c) => c.stateType === "completed").length };

  return detail;
}

export async function getMembers(teamId: string): Promise<LinearMember[]> {
  return mapMembers(await linearFetch(MEMBERS_QUERY, { teamId }, true));
}

export async function getLabels(teamId: string): Promise<KanbanLabel[]> {
  return mapLabels(await linearFetch(LABELS_QUERY, { teamId }, true));
}
