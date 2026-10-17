import { constants, lstatSync, realpathSync, type Stats } from "node:fs";
import { open, type FileHandle } from "node:fs/promises";
import { basename } from "node:path";
import { resolveExisting, type Root } from "./files";

export type DownloadFile = {
  body: ReadableStream<Uint8Array>;
  size: number;
  disposition: string;
};

const CHUNK = 64 * 1024;
const OPEN_READ = constants.O_RDONLY | constants.O_NOFOLLOW | constants.O_NONBLOCK;

export function contentDisposition(filename: string): string {
  for (let i = 0; i < filename.length; i++) {
    const code = filename.charCodeAt(i);
    if (code < 0x20 || code === 0x7f) throw new Error("invalid filename");
  }
  let fallback = "";
  for (let i = 0; i < filename.length; i++) {
    const code = filename.charCodeAt(i);
    const ch = filename[i];
    fallback += code > 0x7e || ch === '"' || ch === "\\" ? "_" : ch;
  }
  const encoded = encodeURIComponent(filename).replace(/['()*]/g, (ch) =>
    `%${ch.charCodeAt(0).toString(16).toUpperCase()}`,
  );
  return `attachment; filename="${fallback}"; filename*=UTF-8''${encoded}`;
}

export async function downloadFile(root: Root, rel: string): Promise<DownloadFile> {
  return downloadFileAt(resolveExisting(root, rel));
}

// Shared open+verify core (O_NOFOLLOW open, lstat match, realpath match) — extracted so the new
// Range-capable raw path (openRawStreamAt) gets the SAME safety checks as a download, plus its own
// cap check, without duplicating them.
async function openVerifiedAt(absPath: string): Promise<{ handle: FileHandle; stat: Stats }> {
  const handle = await open(absPath, OPEN_READ);
  try {
    const stat = await handle.stat();
    if (!stat.isFile()) throw new Error("not a regular file");
    const listed = lstatSync(absPath);
    if (!listed.isFile() || listed.dev !== stat.dev || listed.ino !== stat.ino)
      throw new Error("changed on disk");
    if (realpathSync(absPath) !== absPath) throw new Error("changed on disk");
    return { handle, stat };
  } catch (error) {
    await handle.close().catch(() => undefined);
    throw error;
  }
}

export async function downloadFileAt(absPath: string): Promise<DownloadFile> {
  const { handle, stat } = await openVerifiedAt(absPath);
  let disposition: string;
  try {
    disposition = contentDisposition(basename(absPath));
  } catch (error) {
    await handle.close().catch(() => undefined);
    throw error;
  }
  return { body: streamHandle(handle, stat.size), size: stat.size, disposition };
}

// New for raw's Range-capable inline path (pdf/audio/video, spec §7/§10): same safety checks as a
// download, plus a cap check BEFORE any byte streams — closes the handle and throws "too large"
// rather than ever starting a response that would only be refused partway through.
export async function openRawStreamAt(absPath: string, cap: number): Promise<{ handle: FileHandle; size: number }> {
  const { handle, stat } = await openVerifiedAt(absPath);
  if (stat.size > cap) {
    await handle.close().catch(() => undefined);
    throw new Error("too large");
  }
  return { handle, size: stat.size };
}

export type RangeSpec = { start: number; end: number };

// range, when given, streams only [start, end] (inclusive) instead of the whole file — the ONLY
// change from the original whole-file behavior. Every existing caller (downloadFileAt) passes no
// range and streams the whole file exactly as before; the rest of this function is unchanged, since
// it operates generically on offset/remaining regardless of their initial values.
export function streamHandle(handle: FileHandle, size: number, range?: RangeSpec): ReadableStream<Uint8Array> {
  let offset = range ? range.start : 0;
  let remaining = range ? range.end - range.start + 1 : size;
  let canceled = false;
  let pendingRead: Promise<unknown> | null = null;
  let closed: Promise<void> | undefined;
  const closeOnce = () => {
    closed ??= handle.close().then(() => undefined, () => undefined);
    return closed;
  };
  return new ReadableStream<Uint8Array>({
    async pull(controller) {
      if (canceled) return;
      if (remaining === 0) {
        controller.close();
        await closeOnce();
        return;
      }
      const want = Math.min(CHUNK, remaining);
      const buf = new Uint8Array(want);
      const read = handle.read(buf, 0, want, offset);
      pendingRead = read;
      try {
        const { bytesRead } = await read;
        pendingRead = null;
        if (canceled) return;
        if (bytesRead === 0) {
          controller.error(new Error("truncated"));
          await closeOnce();
          return;
        }
        offset += bytesRead;
        remaining -= bytesRead;
        controller.enqueue(bytesRead === want ? buf : buf.subarray(0, bytesRead));
        if (remaining === 0) {
          controller.close();
          await closeOnce();
        }
      } catch (error) {
        pendingRead = null;
        if (canceled) return;
        controller.error(error);
        await closeOnce();
      }
    },
    async cancel() {
      canceled = true;
      if (pendingRead) await pendingRead.then(() => undefined, () => undefined);
      await closeOnce();
    },
  }, { highWaterMark: 0 });
}

// Parses a `Range: bytes=...` header against a known total `size`. Single-range only (no
// "bytes=0-99,200-299" support — ponytail: browsers seeking audio/video/pdf send one range at a
// time; add multi-range only if a real client ever needs it). Returns null for "no Range header AT
// ALL" (caller sends a full 200, unchanged for every existing image caller) — an explicitly EMPTY
// header ("") is a different case and falls through to "unsatisfiable" below, same as any other
// malformed value (cold review round 1 F3: `!header` conflated the two, since "" is also falsy).
export function parseRange(header: string | null, size: number): RangeSpec | "unsatisfiable" | null {
  if (header === null) return null;
  const match = /^bytes=(\d*)-(\d*)$/.exec(header.trim());
  if (!match || (match[1] === "" && match[2] === "")) return "unsatisfiable";
  let start: number;
  let end: number;
  if (match[1] === "") {
    const suffixLength = Number(match[2]);
    if (!Number.isFinite(suffixLength) || suffixLength <= 0) return "unsatisfiable";
    start = Math.max(0, size - suffixLength);
    end = size - 1;
  } else {
    start = Number(match[1]);
    end = match[2] === "" ? size - 1 : Number(match[2]);
  }
  if (!Number.isFinite(start) || !Number.isFinite(end) || start < 0 || start >= size || start > end)
    return "unsatisfiable";
  return { start, end: Math.min(end, size - 1) };
}
