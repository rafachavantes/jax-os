import { collectorResponse } from "@/server/api";
import { getSubscriptions } from "@/server/collectors/subscriptions";
import { readGeneralSettings } from "@/server/settings";

export const dynamic = "force-dynamic";

export async function GET() {
  // Fail closed: an unreadable/malformed settings file means every provider is off.
  const settings = readGeneralSettings();
  const agents = settings.ok ? settings.data.integrations.agents : null;
  // MOA-504 D6: the meters follow the agent switches; the collector keeps its own flag names.
  const enabled = agents
    ? { claude: agents.claude, codex: agents.codex, opencodeGo: agents.opencode }
    : { claude: false, codex: false, opencodeGo: false };
  return collectorResponse(() => getSubscriptions(enabled));
}
