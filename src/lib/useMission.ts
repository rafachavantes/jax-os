"use client";

import { useQuery } from "@tanstack/react-query";
import { fetchEnvelope } from "./api";

export type MissionEndpoint = "projects" | "agents" | "activity" | "alerts" | "hub" | "prs" | "worktrees";

// `query` is a pre-built, non-interpolated query string (e.g. "scope=mine") that is part of the
// cache key, so one scope's result can never flash as another's (spec "Relevant PR scope").
export function missionQueryKey(endpoint: MissionEndpoint, query: string | null = null) {
  return ["mission", endpoint, query] as const;
}

export function useMission<T>(endpoint: MissionEndpoint, refetchInterval: number, enabled = true, query: string | null = null) {
  return useQuery({
    queryKey: missionQueryKey(endpoint, query),
    queryFn: ({ signal }) => fetchEnvelope<T>(`/api/mission/${endpoint}${query ? `?${query}` : ""}`, signal),
    refetchInterval,
    enabled,
  });
}
