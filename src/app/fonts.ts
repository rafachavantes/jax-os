import { GeistSans } from "geist/font/sans";
import { GeistMono } from "geist/font/mono";
import localFont from "next/font/local";

// ponytail: Quicksand (--font-logo) intentionally not loaded — no component renders the
// wordmark as text any more (the brand lockup is an SVG); add via next/font/local if one does.
// Orbitron (OFL 1.1, wght 400-900) is ONE variable file; its licence ships beside it in src/fonts/.
const orbitron = localFont({
  src: [{ path: "../fonts/Orbitron-VariableFont_wght.ttf", weight: "400 900", style: "normal" }],
  variable: "--font-orbitron",
});

export const fontVariables = [
  GeistSans.variable,
  GeistMono.variable,
  orbitron.variable,
].join(" ");
