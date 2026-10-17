export const LOCAL_OPTS = { encoding: "utf8" as const, timeout: 5_000, maxBuffer: 1024 * 1024 };
export const SMALL_JSON_CAP = 64 * 1024;
export const WRITE_JSON_CAP = 16 * 1024 * 1024;
export const COMMENT_JSON_CAP = 256 * 1024;
export const COMMENT_CAP = 32 * 1024;
export function boundedString(value: unknown, cap: number, allowEmpty = false): value is string {
  return typeof value === "string" && (allowEmpty || value.length > 0)
    && Buffer.byteLength(value, "utf8") <= cap;
}
export function validRel(value: unknown, allowEmpty = false): value is string {
  return boundedString(value, 4096, allowEmpty) && !value.includes("\0");
}
export function validBasename(value: unknown): value is string {
  return boundedString(value, 255) && !value.includes("/")
    && !value.includes("..") && !value.includes("\0");
}
