// Copies monaco-editor's min/vs bundle, plus its LICENSE and
// ThirdPartyNotices.txt, into public/ so the editor loads same-origin
// (loopback) with zero external network — no CDN.
// Clears dest first so a monaco version bump can't leave stale files behind.
// Runs as postinstall + prebuild; ~16 MB, gitignored, regenerated from the dep.
import { cp, rm, access } from "node:fs/promises";
import { fileURLToPath } from "node:url";

const src = fileURLToPath(new URL("../node_modules/monaco-editor/min/vs", import.meta.url));
const dest = fileURLToPath(new URL("../public/monaco-vs/vs", import.meta.url));

try {
  await access(src);
} catch {
  // monaco not installed yet (e.g. postinstall ordering) — skip quietly.
  console.log("[copy-monaco] monaco-editor not found, skipping");
  process.exit(0);
}

await rm(dest, { recursive: true, force: true }); // drop stale files from a previous monaco version
await cp(src, dest, { recursive: true });
console.log(`[copy-monaco] copied min/vs -> public/monaco-vs/vs`);

for (const name of ["LICENSE", "ThirdPartyNotices.txt"]) {
  const srcFile = fileURLToPath(new URL(`../node_modules/monaco-editor/${name}`, import.meta.url));
  const destFile = fileURLToPath(new URL(`../public/monaco-vs/${name}`, import.meta.url));
  try {
    await access(srcFile);
  } catch {
    // older monaco-editor without this file — skip quietly, never fail the install.
    console.log(`[copy-monaco] ${name} not found, skipping`);
    continue;
  }
  await rm(destFile, { force: true }); // drop a stale copy from a previous monaco version
  await cp(srcFile, destFile);
  console.log(`[copy-monaco] copied ${name} -> public/monaco-vs/${name}`);
}
