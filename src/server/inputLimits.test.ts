import { describe, expect, it } from "vitest";
import {
  COMMENT_CAP,
  COMMENT_JSON_CAP,
  SMALL_JSON_CAP,
  WRITE_JSON_CAP,
  boundedString,
  validBasename,
  validRel,
} from "./inputLimits";

describe("caps", () => {
  it("uses the I2 body and comment limits", () => {
    expect(SMALL_JSON_CAP).toBe(64 * 1024);
    expect(WRITE_JSON_CAP).toBe(16 * 1024 * 1024);
    expect(COMMENT_JSON_CAP).toBe(256 * 1024);
    expect(COMMENT_CAP).toBe(32 * 1024);
  });
});

describe("boundedString", () => {
  it("accepts ASCII at the cap and rejects one byte over", () => {
    expect(boundedString("abcd", 4)).toBe(true);
    expect(boundedString("abcde", 4)).toBe(false);
  });
  it("counts UTF-8 bytes, not characters", () => {
    expect(boundedString("é", 2)).toBe(true);
    expect(boundedString("é", 1)).toBe(false);
  });
  it("rejects empty unless allowEmpty is set", () => {
    expect(boundedString("", 10)).toBe(false);
    expect(boundedString("", 10, true)).toBe(true);
  });
  it("rejects non-strings", () => {
    expect(boundedString(1, 10)).toBe(false);
    expect(boundedString(null, 10)).toBe(false);
    expect(boundedString(undefined, 10)).toBe(false);
    expect(boundedString({}, 10)).toBe(false);
    expect(boundedString([], 10)).toBe(false);
  });
  it("accepts COMMENT_CAP bytes and rejects one byte over", () => {
    expect(boundedString("é".repeat(COMMENT_CAP / 2), COMMENT_CAP)).toBe(true);
    expect(boundedString("é".repeat(COMMENT_CAP / 2) + "x", COMMENT_CAP)).toBe(false);
  });
});

describe("validRel", () => {
  it("accepts 4096 UTF-8 bytes and rejects 4097", () => {
    expect(validRel("a".repeat(4096))).toBe(true);
    expect(validRel("a".repeat(4097))).toBe(false);
    expect(validRel("é".repeat(2048))).toBe(true);
    expect(validRel("é".repeat(2048) + "x")).toBe(false);
  });
  it("rejects empty unless allowEmpty is set", () => {
    expect(validRel("")).toBe(false);
    expect(validRel("", true)).toBe(true);
  });
  it("rejects NUL", () => {
    expect(validRel("a\0b")).toBe(false);
  });
  it("rejects non-strings", () => {
    expect(validRel(1)).toBe(false);
  });
});

describe("validBasename", () => {
  it("accepts 255 UTF-8 bytes and rejects 256", () => {
    expect(validBasename("a".repeat(255))).toBe(true);
    expect(validBasename("a".repeat(256))).toBe(false);
    expect(validBasename("é".repeat(127) + "a")).toBe(true);
    expect(validBasename("é".repeat(128))).toBe(false);
  });
  it("rejects empty, slash, traversal, and NUL", () => {
    expect(validBasename("")).toBe(false);
    expect(validBasename("a/b")).toBe(false);
    expect(validBasename("..")).toBe(false);
    expect(validBasename("foo..bar")).toBe(false);
    expect(validBasename("a\0b")).toBe(false);
  });
  it("accepts ordinary dotted names", () => {
    expect(validBasename("file.txt")).toBe(true);
    expect(validBasename(".gitignore")).toBe(true);
  });
});
