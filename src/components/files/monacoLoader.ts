// Client-only. Points @monaco-editor/react at our self-hosted bundle
// (public/monaco-vs/vs, served same-origin over loopback) instead of the
// default jsdelivr CDN. Zero external network. Runs at module-eval time so
// config precedes any loader.init() — the editor module imports this before
// mounting <Editor>. paths.vs MUST match scripts/copy-monaco.mjs dest.
import { loader } from "@monaco-editor/react";

loader.config({ paths: { vs: "/monaco-vs/vs" } });
