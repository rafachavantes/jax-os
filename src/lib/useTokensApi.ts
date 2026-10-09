"use client";

import { useQuery } from "@tanstack/react-query";
import { fetchEnvelope } from "./api";

export function useTokensApi<T>(path: string, refetchInterval: number, enabled = true) {
  return useQuery({
    queryKey: ["tokens", path],
    queryFn: ({ signal }) => fetchEnvelope<T>(`/api/tokens/${path}`, signal),
    refetchInterval,
    enabled,
  });
}
