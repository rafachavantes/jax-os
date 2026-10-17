import { getDb } from "../../../../server/db";
import { getActiveMission } from "../../../../server/db/missions";
import { collectorResponse } from "../../../../server/api";

export type MissionCurrentRouteDeps = { getDb: typeof getDb; getActiveMission: typeof getActiveMission };
const systemDeps: MissionCurrentRouteDeps = { getDb, getActiveMission };

export async function handleMissionCurrent(_req?: Request, deps: MissionCurrentRouteDeps = systemDeps) {
  return collectorResponse(() => deps.getActiveMission(deps.getDb()));
}
