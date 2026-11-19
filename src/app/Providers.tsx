"use client";

import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { useState } from "react";
import { AnswerDraftsProvider } from "@/components/mission/AnswerDrafts";
import { FilesWorkspaceProvider } from "@/components/files/FilesWorkspaceProvider";
import { ToolsCredentialHost } from "@/components/tools/ProviderCredentialDialog";
import { ToolsDraftProvider } from "@/components/tools/ToolsDraftProvider";

export function Providers({ children }: { children: React.ReactNode }) {
  const [client] = useState(() => new QueryClient());
  return (
    <QueryClientProvider client={client}>
      <FilesWorkspaceProvider>
        <ToolsDraftProvider>
          {/* Hoisted above route children (M3b): an unfinished answer survives a route unmount. */}
          <AnswerDraftsProvider>
            {children}
          </AnswerDraftsProvider>
          <ToolsCredentialHost />
        </ToolsDraftProvider>
      </FilesWorkspaceProvider>
    </QueryClientProvider>
  );
}
