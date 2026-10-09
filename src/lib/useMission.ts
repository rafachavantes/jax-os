"use client";

import { useQuery } from "@tanstack/react-query";
import { fetchEnvelope } from "./api";

export type MissionEndpoint = "projects" | "agents" | "activity" | "alerts" | "hub" | "prs" | "worktrees";

export function useMission<T>(endpoint: MissionEndpoint, refetchInterval: number, enabled = true) {
  return useQuery({
    queryKey: ["mission", endpoint],
    queryFn: ({ signal }) => fetchEnvelope<T>(`/api/mission/${endpoint}`, signal),
    refetchInterval,
    enabled,
  });
}
