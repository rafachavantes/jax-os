// MOA-504 D4: seed settings.json once per process start, before Next serves any request.
// Node runtime only, never during `next build`. Python never seeds (spec D4).
export async function register(): Promise<void> {
  if (process.env.NEXT_RUNTIME !== "nodejs" || process.env.NEXT_PHASE === "phase-production-build") return;
  try {
    const { getDb } = await import("./server/db");
    const { seedAgentsIfFirstRun } = await import("./server/settings");
    const { isCliPresent } = await import("./server/collectors/cli-presence");
    seedAgentsIfFirstRun(getDb(), (name) => isCliPresent(name, process.env));
  } catch (err) {
    // A failure before the file is replaced leaves no file: defaults apply (all off + banner).
    // An audit-insert failure after it keeps the seeded file (the file is the marker): log only.
    const afterWrite = err instanceof Error && err.name === "SettingsAuditFailedError";
    console.error(afterWrite ? "[jaxos] agent seed: settings.json written, audit row failed" : "[jaxos] agent seed failed, defaults apply", err);
  }
}
