"use client";

import { useQuery } from "@tanstack/react-query";
import type { Envelope } from "./api";
import { fetchEnvelope } from "./api";

export function useTokensApi<T>(
  path: string,
  refetchInterval: number | ((envelope: { ok: true; data: T } | undefined) => number),
  enabled = true,
) {
  return useQuery({
    queryKey: ["tokens", path],
    queryFn: ({ signal }) => fetchEnvelope<T>(`/api/tokens/${path}`, signal),
    refetchInterval:
      typeof refetchInterval === "number"
        ? refetchInterval
        : (query) => {
            const data = query.state.data as Envelope<T> | undefined;
            return data?.ok ? refetchInterval(data) : 60_000;
          },
    enabled,
  });
}
