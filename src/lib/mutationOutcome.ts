export class MutationRejected extends Error {
  constructor(message: string, readonly status = 200) { super(message); }
}
export type MutationFailure = {
  ok: false;
  code: "audit-unavailable" | "mutation-rejected" | "mutation-unconfirmed" | "audit-finalization-failed";
  effect: "not-applied" | "unconfirmed" | "applied";
  audit: "unavailable" | "pending" | "recorded";
  error: string;
};
