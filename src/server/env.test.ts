import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { homedir } from "node:os";
import { join } from "node:path";
import { apiBaseUrl, jaxosHome } from "./env";

beforeEach(() => {
  delete process.env.JAXOS_HOME;
  delete process.env.PORT;
});

afterEach(() => {
  delete process.env.JAXOS_HOME;
  delete process.env.PORT;
});

describe("jaxosHome / apiBaseUrl (spec: JAXOS_HOME / base URL resolution)", () => {
  it("jaxosHome defaults to ~/.jax-os with JAXOS_HOME unset (default-equals-today)", () => {
    expect(jaxosHome()).toBe(join(homedir(), ".jax-os"));
  });
  it("jaxosHome uses JAXOS_HOME verbatim, read fresh on every call (no caching)", () => {
    expect(jaxosHome()).toBe(join(homedir(), ".jax-os"));
    process.env.JAXOS_HOME = "/tmp/y";
    expect(jaxosHome()).toBe("/tmp/y");
  });
  it("apiBaseUrl defaults to 127.0.0.1:3100 with PORT unset (default-equals-today)", () => {
    expect(apiBaseUrl()).toBe("http://127.0.0.1:3100");
  });
  it("apiBaseUrl uses PORT when set", () => {
    process.env.PORT = "9999";
    expect(apiBaseUrl()).toBe("http://127.0.0.1:9999");
  });
});
