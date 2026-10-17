"use client";

import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { useState } from "react";
import { FilesWorkspaceProvider } from "@/components/files/FilesWorkspaceProvider";
import { ToolsCredentialHost } from "@/components/tools/ProviderCredentialDialog";
import { ToolsDraftProvider } from "@/components/tools/ToolsDraftProvider";

export function Providers({ children }: { children: React.ReactNode }) {
  const [client] = useState(() => new QueryClient());
  return (
    <QueryClientProvider client={client}>
      <FilesWorkspaceProvider>
        <ToolsDraftProvider>
          {children}
          <ToolsCredentialHost />
        </ToolsDraftProvider>
      </FilesWorkspaceProvider>
    </QueryClientProvider>
  );
}
