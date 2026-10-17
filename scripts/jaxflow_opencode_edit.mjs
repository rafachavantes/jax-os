#!/usr/bin/env node
import { createRequire } from "node:module";

const require = createRequire(import.meta.url);
const { applyEdits, modify, parse, parseTree } = require("jsonc-parser");

const IO_CAP = 16 * 1024 * 1024;
const SOURCE_CAP = 1024 * 1024;

let done = false;

function fail(error) {
  if (done) return;
  done = true;
  process.stdout.write(JSON.stringify({ ok: false, error }));
  process.exit(0);
}

function hasDuplicateKeys(node) {
  if (!node?.children) return false;
  if (node.type === "object") {
    const seen = new Set();
    for (const property of node.children) {
      const key = property.children?.[0]?.value;
      if (key !== undefined) {
        if (seen.has(key)) return true;
        seen.add(key);
      }
      if (hasDuplicateKeys(property.children?.[1])) return true;
    }
    return false;
  }
  return node.children.some(hasDuplicateKeys);
}

function checked(text, jsonc) {
  const options = { allowTrailingComma: jsonc, disallowComments: !jsonc };
  const errors = [];
  const tree = parseTree(text, errors, options);
  if (errors.length > 0 || !tree || tree.type !== "object" || hasDuplicateKeys(tree)) return null;
  const valueErrors = [];
  const value = parse(text, valueErrors, options);
  if (valueErrors.length > 0 || !value || Array.isArray(value) || typeof value !== "object") return null;
  return value;
}

const chunks = [];
let size = 0;
process.stdin.on("data", (chunk) => {
  size += chunk.length;
  if (size > IO_CAP) {
    process.stdin.destroy();
    fail("agent-settings-too-large");
    return;
  }
  chunks.push(chunk);
});
process.stdin.on("end", () => {
  if (done) return;
  let payload;
  try {
    payload = JSON.parse(Buffer.concat(chunks).toString("utf8"));
  } catch {
    fail("native-config-malformed");
    return;
  }
  if (!payload || typeof payload !== "object" || typeof payload.source !== "string" || !Array.isArray(payload.edits)) {
    fail("native-config-malformed");
    return;
  }
  if (Buffer.byteLength(payload.source) > SOURCE_CAP || payload.edits.length > 64) {
    fail(payload.edits.length > 64 ? "native-config-malformed" : "agent-settings-too-large");
    return;
  }
  const jsonc = payload.jsonc === true;
  if (!checked(payload.source, jsonc)) {
    fail("native-config-malformed");
    return;
  }
  const eol = payload.source.includes("\r\n") ? "\r\n" : "\n";
  let text = payload.source;
  for (const edit of payload.edits) {
    if (!edit || typeof edit !== "object" || !Array.isArray(edit.path)) {
      fail("native-config-malformed");
      return;
    }
    const value = edit.remove === true ? undefined : edit.value;
    text = applyEdits(text, modify(text, edit.path, value, {
      formattingOptions: { insertSpaces: true, tabSize: 2, eol },
    }));
    if (!checked(text, jsonc)) {
      fail("native-config-malformed");
      return;
    }
  }
  const value = checked(text, jsonc);
  done = true;
  process.stdout.write(JSON.stringify({ ok: true, text, value }));
});
