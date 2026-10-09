import { collectorResponse } from "@/server/api";
import { getDb } from "@/server/db";
import { getInventoryData, systemInventoryDeps } from "@/server/collectors/inventory";
import { getToolsData } from "@/server/collectors/tools";
import { joinInventoryOperations, restoreManagedCapabilities } from "@/server/db/inventory-operations";

export const dynamic = "force-dynamic";

// Integrated Tools read (spec §7): the native inventory snapshot is the single
// discovery. It is projected into the per-executor plugin/skill/agent arrays
// AND served as the executors/recovery contract the Inventory UI consumes, so
// no second complete discovery runs. `/api/tools/inventory` stays the finite
// action + operation-lookup endpoint.
export async function GET() {
  return collectorResponse(async () => {
    const db = getDb();
    const data = await getInventoryData(systemInventoryDeps());
    const integrated = joinInventoryOperations(db, restoreManagedCapabilities(db, data));
    return { ...integrated, ...getToolsData(integrated) };
  });
}
