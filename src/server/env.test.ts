import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { homedir } from "node:os";
import { join } from "node:path";
import { jaxosHome } from "./env";

beforeEach(() => {
  delete process.env.JAXOS_HOME;
});

afterEach(() => {
  delete process.env.JAXOS_HOME;
});

describe("jaxosHome (spec: JAXOS_HOME resolution)", () => {
  it("jaxosHome defaults to ~/.jax-os with JAXOS_HOME unset (default-equals-today)", () => {
    expect(jaxosHome()).toBe(join(homedir(), ".jax-os"));
  });
  it("jaxosHome uses JAXOS_HOME verbatim, read fresh on every call (no caching)", () => {
    expect(jaxosHome()).toBe(join(homedir(), ".jax-os"));
    process.env.JAXOS_HOME = "/tmp/y";
    expect(jaxosHome()).toBe("/tmp/y");
  });
});
