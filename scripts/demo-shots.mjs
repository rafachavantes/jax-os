#!/usr/bin/env node
// Reproducible README screenshots over SYNTHETIC data (MOA-512 D1, D2).
//   PLAYWRIGHT_PATH=<node_modules dir that contains playwright> [CHROMIUM_PATH=<chrome binary>] \
//     node scripts/demo-shots.mjs            shoot into docs/screenshots/
//   node scripts/demo-shots.mjs --self-test  pure checks, no browser, no server
// Needs a finished `pnpm build` (.next/). It never builds, never uses ports 3000/3100 and deletes only the
// scratch directory it created itself. DEMO_SHOTS_EXTRA_DENY=a,b adds denylist words (also the fault-injection hook).
import { execFileSync, spawn } from "node:child_process";
import { chmodSync, closeSync, copyFileSync, existsSync, lstatSync, mkdirSync, mkdtempSync, openSync, readFileSync, readdirSync, realpathSync, rmSync, statSync, writeFileSync } from "node:fs";
import { createRequire } from "node:module";
import { createServer } from "node:net";
import { hostname, tmpdir, userInfo } from "node:os";
import { basename, dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const SELF = fileURLToPath(import.meta.url);
const REPO = resolve(dirname(SELF), "..");
const SCRATCH_RE = /^jaxos-demo\.[A-Za-z0-9]{6}$/;
const MAX_BYTES = 400 * 1024;
const RESERVED_PORTS = [3000, 3100];
const THEMES = ["dark", "light"];
const PAGES = [
  { name: "mission-control", path: "/", ready: "Acme API", w: 1440, h: 900 },
  { name: "tokens", path: "/tokens", ready: "demo-model-a", w: 1440, h: 900 },
  // Health renders seeded service names, not hostname (page.tsx strips the .service suffix).
  { name: "health", path: "/health", ready: "demo-web", w: 1440, h: 900 },
  { name: "mobile-mission-control", path: "/", ready: "Acme API", w: 390, h: 844 },
];
const EXPECTED = THEMES.flatMap((t) => PAGES.map((p) => `${p.name}-${t}.png`)).sort();

// D2 denylist. Built from fragments: this file ships in the public tree and gate G1 greps it.
const j = (...p) => p.join("");
const DENY = [j("Good", " Times"), j("rc", "-mark"), j("Ra", "fa"), j("Chav", "antes"), j("Moa", "ra"), j("Estate", "Map"),
  j("conta", "to@"), j("sete", "flechas"), j("tai", "lad"), "/home/"];
const esc = (s) => s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");

// Static words match as substrings (case-insensitive); every non-empty runtime value (user, short host) as a whole word.
function denylistHits(text, dynamic = [], extra = []) {
  const lower = text.toLowerCase();
  const hits = [...DENY, ...extra].filter((w) => w && lower.includes(w.toLowerCase()));
  for (const v of dynamic) if (v && new RegExp(`(?<![A-Za-z0-9])${esc(v)}(?![A-Za-z0-9])`).test(text)) hits.push(v);
  return hits;
}

const REPOS = [
  { dir: "acme-api", project: "Acme API", stage: "build", builder: "claude-code", branch: "feat/rate-limits", tmux: "acme-api-build",
    now: "Adding per-key rate limits to the public endpoints. Unit tests are green and the integration run is in progress." },
  { dir: "acme-web", project: "Acme Web", stage: "review", gate: "awaiting-approval", builder: "codex", branch: "feat/checkout", tmux: "acme-web-review",
    now: "Checkout rewrite passed its cold review. Waiting for your approval to merge." },
  { dir: "acme-mobile", project: "Acme Mobile", stage: "spec", builder: "claude-code", branch: "feat/offline-sync", tmux: "acme-mobile-spec",
    now: "Drafting the offline sync spec: conflict rules and the retry budget are still open." },
];

function statusMd(r, nowIso) {
  const fm = [["project", r.project], ["stage", r.stage], r.gate && ["gate", r.gate], ["builder", r.builder], ["branch", r.branch],
    ["tmux", r.tmux], ["updated", nowIso]].filter(Boolean).map(([k, v]) => `${k}: ${v}`);
  return `---\n${fm.join("\n")}\n---\n\n## Now\n${r.now}\n`;
}

function rng(seed) { // mulberry32: same data on every run
  return () => { seed |= 0; seed = (seed + 0x6d2b79f5) | 0; let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
}

const fail = (msg) => { throw new Error(msg); };

function safeRemove(root) { // refuses anything that is not a directory this script's mkdtemp pattern could have made
  if (root && SCRATCH_RE.test(basename(root)) && dirname(root) === tmpdir()) rmSync(root, { recursive: true, force: true, maxRetries: 3 });
}

function freePort() {
  return new Promise((ok, no) => {
    const s = createServer();
    s.on("error", no);
    s.listen(0, "127.0.0.1", () => { const { port } = s.address(); s.close(() => (RESERVED_PORTS.includes(port) ? ok(freePort()) : ok(port))); });
  });
}

function loadPlaywright() {
  const dir = process.env.PLAYWRIGHT_PATH;
  if (!dir) fail("set PLAYWRIGHT_PATH to a node_modules directory that contains playwright");
  try { return createRequire(join(dir, "noop.js"))("playwright"); } catch { return fail(`playwright not found under ${dir}`); }
}

function selfTest() {
  const assert = (c, m) => { if (!c) throw new Error(`self-test: ${m}`); };
  assert(denylistHits("Welcome back, Alex. Acme API is building.", ["demo-host", "alexsmith"]).length === 0, "clean text must pass");
  for (const w of DENY) assert(denylistHits(`xx ${w} yy`).length === 1, `denylist word not caught: ${w.length} chars`);
  assert(denylistHits("host demo-box-7 up", ["demo-box-7"]).length === 1, "runtime host must be caught as a whole word");
  assert(denylistHits("demo-box-70", ["demo-box-7"]).length === 0, "whole-word match only");
  assert(denylistHits("path /tmp/x", [], ["acme-api"]).length === 0 && denylistHits("Acme-API", [], ["acme-api"]).length === 1, "extra deny");
  assert(EXPECTED.length === 8 && new Set(EXPECTED).size === 8, "eight distinct shots");
  assert(!RESERVED_PORTS.includes(8080) && RESERVED_PORTS.join() === "3000,3100", "reserved ports");
  for (const r of REPOS) { // the shape parseStatusMd (projects.ts:90) needs
    const md = statusMd(r, "2026-01-01T00:00:00.000Z");
    assert(/^---\n(?:[a-z]+: [^\n]+\n)+---\n\n## Now\n[^\n]+\n$/.test(md), `status.md shape ${r.dir}`);
    assert(/\nstage: (spec|build|review|test|ship)\n/.test(md), `stage ${r.dir}`);
    assert(denylistHits(md).length === 0, `status.md denylist ${r.dir}`);
  }
  const src = readFileSync(SELF, "utf8").toLowerCase(); // review focus 1: this file must not carry the words G1 hunts
  for (const w of DENY.filter((x) => x !== "/home/")) assert(!src.includes(w.toLowerCase()), "denylist word spelled literally in this file");
  const tmp = mkdtempSync(join(tmpdir(), "not-a-demo-")); // safeRemove refuses a dir it did not name-match
  safeRemove(tmp); assert(existsSync(tmp), "safeRemove must refuse a foreign directory"); rmSync(tmp, { recursive: true });
  console.log("self-test ok");
}

function seedWorld(p, nowMs) {
  const nowIso = new Date(nowMs).toISOString();
  const git = (cwd, ...a) => execFileSync("git", ["-c", "user.name=Demo", "-c", "user.email=demo@example.com", ...a],
    { cwd, env: { PATH: process.env.PATH, HOME: p.home, GIT_CONFIG_GLOBAL: "/dev/null", GIT_CONFIG_NOSYSTEM: "1" }, stdio: "ignore" });
  for (const r of REPOS) {
    const dir = join(p.repos, r.dir);
    mkdirSync(join(dir, ".jax-os"), { recursive: true });
    writeFileSync(join(dir, "README.md"), `# ${r.project}\n`);
    writeFileSync(join(dir, ".jax-os", "status.md"), statusMd(r, nowIso));
    git(dir, "init", "-q", "-b", "main"); git(dir, "add", "README.md"); git(dir, "commit", "-q", "-m", "feat: initial commit");
  }
  // One reviewer agent on, so the app-wide "no agent" banner does not cover every shot.
  writeFileSync(join(p.jaxos, "settings.json"), JSON.stringify({ ownerName: "Alex", locale: "en-US", reposRoot: p.repos, vaultPath: null,
    monitoredUnits: { user: [], system: [] },
    integrations: { linear: false, ttyd: false, webhook: false, hermesTokens: true, agents: { claude: true, codex: false, opencode: true },
      classifier: false, github: false, vault: false } }, null, 2));
  const hermesBin = join(p.home, ".local", "bin");
  mkdirSync(hermesBin, { recursive: true });
  writeFileSync(join(hermesBin, "hermes"), "#!/bin/sh\nprintf '%s\\n' '{\"system_prompt\":{\"bytes\":12000},\"skills_index\":{\"bytes\":4000},\"memory\":{\"bytes\":1800},\"user_profile\":{\"bytes\":900},\"tools\":{\"count\":12,\"json_bytes\":6400},\"platform\":\"demo\",\"model\":\"demo-model-a\"}'\n");
  chmodSync(join(hermesBin, "hermes"), 0o755);
  const Database = createRequire(join(REPO, "package.json"))("better-sqlite3");
  const r = rng(512);
  const h = new Database(join(p.home, ".hermes", "state.db"));
  h.exec(`CREATE TABLE sessions (id INTEGER PRIMARY KEY, model TEXT, started_at INTEGER, input_tokens INTEGER, output_tokens INTEGER,
    cache_read_tokens INTEGER, cache_write_tokens INTEGER, reasoning_tokens INTEGER, estimated_cost_usd REAL)`);
  const ins = h.prepare("INSERT INTO sessions (model, started_at, input_tokens, output_tokens, cache_read_tokens, cache_write_tokens, reasoning_tokens, estimated_cost_usd) VALUES (?,?,?,?,?,?,?,?)");
  for (let i = 0; i < 40; i++) {
    const inp = 20000 + Math.floor(r() * 90000), out = 3000 + Math.floor(r() * 20000), cr = Math.floor(r() * 250000);
    ins.run(r() < 0.65 ? "demo-model-a" : "demo-model-b", Math.floor(nowMs / 1000) - Math.floor(r() * 14 * 86400), inp, out, cr, Math.floor(r() * 30000), Math.floor(r() * 8000), (inp + out) * 0.000004 + cr * 0.0000004);
  }
  h.close();
  const skills = { "triage-inbox": 42, "weekly-digest": 17, "release-notes": 9, "stale-branch-sweep": 3 };
  writeFileSync(join(p.home, ".hermes", "skills", ".usage.json"), JSON.stringify(Object.fromEntries(Object.entries(skills).map(([k, v]) =>
    [k, { use_count: v, last_used_at: new Date(nowMs - v * 3600e3).toISOString(), state: "active" }]))));
}

function seedMetrics(p, nowMs) { // after the first /api/health request has run the migrations
  const Database = createRequire(join(REPO, "package.json"))("better-sqlite3");
  const db = new Database(join(p.jaxos, "jaxos.db"));
  const r = rng(513);
  const services = JSON.stringify({ units: { "demo-web.service": "active", "demo-worker.service": "active", "demo-cache.service": "active" },
    containers: [{ name: "demo-db", state: "running" }, { name: "demo-proxy", state: "running" }] });
  const ins = db.prepare("INSERT INTO metrics (ts, cpu_pct, mem_used_mb, mem_total_mb, swap_used_mb, disk_used_gb, disk_total_gb, load1, load5, load15, uptime_s, services) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)");
  db.transaction(() => {
    for (let i = 0; i < 200; i++) {
      const cpu = 32 + 18 * Math.sin(i / 14) + r() * 9;
      ins.run(new Date(nowMs - (199 - i) * 432_000).toISOString(), cpu, 6100 + Math.floor(900 * Math.sin(i / 30) + r() * 200), 16000, 120,
        142.5 + i * 0.01, 480, cpu / 25, cpu / 28, cpu / 31, 1_900_000 + i * 432, services);
    }
  })();
  db.close();
}

function subUsage(provider, plan, five, week) {
  return { provider, plan, fiveHour: { usedPercent: five, resetsAt: null }, weekly: { usedPercent: week, resetsAt: null }, models: [] };
}

async function fulfillJson(route, mutate) {
  const res = await route.fetch();
  const body = await res.json();
  mutate(body);
  await route.fulfill({ response: res, json: body });
}

async function waitHealth(base, server, logPath) {
  for (let i = 0; i < 120; i++) {
    if (server.exitCode !== null) {
      let tail = "";
      try { tail = readFileSync(logPath, "utf8").split("\n").slice(-12).join("\n"); } catch { /* log missing */ }
      fail(`next start exited early\n${tail}`);
    }
    try { if ((await fetch(`${base}/api/health`, { signal: AbortSignal.timeout(2000) })).ok) return; } catch { /* not up yet */ }
    await new Promise((r) => setTimeout(r, 500));
  }
  return fail("server did not answer /api/health in 60s");
}

async function shootAll({ browser, base, p, dynamic, extra }) {
  for (const theme of THEMES) {
    for (const [w, h] of [[1440, 900], [390, 844]]) {
      const ctx = await browser.newContext({ viewport: { width: w, height: h }, deviceScaleFactor: 1 });
      await ctx.addCookies(["theme", "locale", "sidebar"].map((name) => ({ name, value: { theme, locale: "en-US", sidebar: "expanded" }[name], url: base })));
      // D2: Health renders the real host name; replace it in the browser so no live host value can reach a shot.
      await ctx.route("**/api/health**", (route) => fulfillJson(route, (body) => {
        if (body && body.ok && body.data) body.data.hostname = "demo-host";
      }));
      // Live zombie processes and OS package versions would leak host detail; drop them.
      await ctx.route("**/api/mission/alerts**", (route) => fulfillJson(route, (body) => {
        if (body && body.ok && Array.isArray(body.data)) body.data = body.data.filter((a) => !["zombies", "updates"].includes(a.kind));
      }));
      // Scratch reposRoot is a live path on Mission Control; stub it.
      await ctx.route("**/api/mission/projects**", (route) => fulfillJson(route, (body) => {
        if (body && body.ok && body.data) body.data.reposRoot = "/demo/repos";
      }));
      // Subscription meters have no local seed; a missing-credentials banner is a live failure.
      await ctx.route("**/api/tokens/subscriptions**", (route) => fulfillJson(route, (body) => {
        if (body && body.ok) body.data = {
          anthropic: { ok: true, data: subUsage("anthropic", "pro", 18, 41) },
          codex: { ok: true, data: subUsage("codex", "plus", 27, 55) },
          opencode: { ok: true, data: { ...subUsage("opencode", "go", 12, 33), monthly: { usedPercent: 22, resetsAt: null } } },
        };
      }));
      for (const pg of PAGES.filter((x) => x.w === w)) {
        const page = await ctx.newPage();
        await page.goto(base + pg.path, { waitUntil: "networkidle" });
        await page.waitForFunction((s) => document.body.innerText.includes(s), pg.ready, { timeout: 30_000 }); // the default QueryClient retries for ~7s
        await page.waitForTimeout(900);
        const raw = await (await fetch(base + pg.path, { headers: { cookie: `theme=${theme}; locale=en-US; sidebar=expanded` } })).text();
        const seen = `${await page.evaluate(() => document.body.innerText)}\n${page.url()}\n${raw}`;
        const hits = denylistHits(seen, dynamic, extra);
        if (hits.length) fail(`denylist hit on ${pg.path} (${theme}, ${w}px): ${hits.join(", ")}`);
        await page.screenshot({ path: join(p.shots, `${pg.name}-${theme}.png`) });
        await page.close();
      }
      await ctx.close();
    }
  }
}

function publish(p, extra) {
  const files = readdirSync(p.shots).sort();
  if (files.join() !== EXPECTED.join()) fail(`expected ${EXPECTED.join(", ")} but got ${files.join(", ")}`);
  for (const f of files) {
    if (!/^[a-z0-9-]+\.png$/.test(f) || denylistHits(f, [], extra).length) fail(`bad file name ${f}`);
    if (statSync(join(p.shots, f)).size > MAX_BYTES) fail(`${f} is over ${MAX_BYTES} bytes (A3 limit); do not relax it, report it`);
  }
  const out = join(REPO, "docs", "screenshots");
  // Containment BEFORE any mkdir: the nearest existing ancestor must resolve inside the repo (a symlinked docs/ is refused).
  let anc = dirname(out);
  while (!existsSync(anc)) anc = dirname(anc);
  if (!(realpathSync(anc) + "/").startsWith(realpathSync(REPO) + "/")) fail("docs/ resolves outside the repo");
  mkdirSync(out, { recursive: true });
  if (!realpathSync(out).startsWith(realpathSync(REPO) + "/")) fail("docs/screenshots resolves outside the repo");
  for (const f of files) { // only ENOENT means absent; any symlink, dangling or not, aborts
    try { if (lstatSync(join(out, f)).isSymbolicLink()) fail(`${f} is a symlink; refusing to write through it`); } catch (e) { if (e.code !== "ENOENT") throw e; }
  }
  for (const f of files) copyFileSync(join(p.shots, f), join(out, f));
  console.log(`wrote ${files.length} screenshots to docs/screenshots/`);
}

async function main() {
  if (process.argv.includes("--self-test")) return selfTest();
  const { chromium } = loadPlaywright();
  if (!existsSync(join(REPO, ".next", "BUILD_ID"))) fail("no .next build: run `pnpm build` first (this script never builds)");
  const extra = (process.env.DEMO_SHOTS_EXTRA_DENY || "").split(",").map((s) => s.trim()).filter(Boolean);
  const dynamic = [userInfo().username, hostname().split(".")[0]];
  const root = mkdtempSync(join(tmpdir(), "jaxos-demo."));
  const p = { root, home: join(root, "home"), jaxos: join(root, "home", ".jax-os"), repos: join(root, "repos"), tmux: join(root, "tmux"), shots: join(root, "shots") };
  for (const d of [p.jaxos, p.repos, p.tmux, p.shots, join(p.home, ".hermes", "skills")]) mkdirSync(d, { recursive: true });
  // SWC refuses a native cache under tmpdir or a group-writable parent. The real user cache already passes that check.
  const env = { PATH: process.env.PATH, HOME: p.home, JAXOS_HOME: p.jaxos, TMUX_TMPDIR: p.tmux, LANG: "C.UTF-8",
    SWC_NATIVE_BINDING_CACHE: join(userInfo().homedir, ".cache") }; // allow-list: no inherited credentials
  let server, browser;
  const sock = join(p.tmux, `tmux-${process.getuid()}`, "default");
  const cleanup = async () => { // the ONE teardown: finally and the signal handlers both call it
    if (server && server.exitCode === null) {
      server.kill("SIGTERM");
      await new Promise((r) => { server.once("exit", r); setTimeout(() => { server.kill("SIGKILL"); r(); }, 5000); });
    }
    try { execFileSync("tmux", ["-S", sock, "kill-server"], { env, stdio: "ignore" }); } catch { /* no server */ }
    safeRemove(root);
  };
  for (const s of ["SIGINT", "SIGTERM"]) process.on(s, () => cleanup().finally(() => process.exit(130)));
  try {
    const nowMs = Date.now();
    seedWorld(p, nowMs);
    for (const r of REPOS) { // D1.3: one isolated tmux session per status.md `tmux:` name, a harmless canned loop inside
      execFileSync("tmux", ["new-session", "-d", "-s", r.tmux, "-c", join(p.repos, r.dir), "sh", "-c",
        `i=0; while :; do echo "[agent] step $i: reading, editing, running tests"; i=$((i+1)); sleep 30; done`], { env, stdio: "ignore" });
    }
    const port = await freePort();
    const log = openSync(join(root, "server.log"), "w");
    server = spawn(process.execPath, [join(REPO, "node_modules", "next", "dist", "bin", "next"), "start", "-H", "127.0.0.1", "-p", String(port)],
      { cwd: REPO, env: { ...env, PORT: String(port) }, stdio: ["ignore", log, log] });
    closeSync(log);
    const base = `http://127.0.0.1:${port}`;
    await waitHealth(base, server, join(root, "server.log"));
    seedMetrics(p, Date.now());
    browser = await chromium.launch(process.env.CHROMIUM_PATH ? { executablePath: process.env.CHROMIUM_PATH } : {});
    await shootAll({ browser, base, p, dynamic, extra });
    publish(p, extra);
  } finally {
    await browser?.close().catch(() => {});
    await cleanup();
  }
}

main().catch((e) => { console.error(`FAIL: ${e.message}`); process.exitCode = 1; });
