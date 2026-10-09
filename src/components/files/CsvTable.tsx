"use client";

import { useMemo, useState } from "react";
import { useTranslations } from "next-intl";

// RFC4180-ish, hand-rolled (ponytail: no csv-parse dependency for one small table view) — quoted
// fields, doubled-quote escaping, CRLF/LF/CR line endings, trailing field with no newline.
export function parseCsv(text: string, delimiter = ","): string[][] {
  const rows: string[][] = [];
  let row: string[] = [];
  let field = "";
  let inQuotes = false;
  for (let i = 0; i < text.length; i++) {
    const c = text[i];
    if (inQuotes) {
      if (c === '"') {
        if (text[i + 1] === '"') { field += '"'; i++; } else inQuotes = false;
      } else field += c;
    } else if (c === '"') inQuotes = true;
    else if (c === delimiter) { row.push(field); field = ""; }
    else if (c === "\n" || c === "\r") {
      if (c === "\r" && text[i + 1] === "\n") i++;
      row.push(field); field = ""; rows.push(row); row = [];
    } else field += c;
  }
  if (field !== "" || row.length > 0) { row.push(field); rows.push(row); }
  return rows;
}

export function sortRows(rows: string[][], col: number, dir: "asc" | "desc"): string[][] {
  const sorted = [...rows].sort((a, b) => (a[col] ?? "").localeCompare(b[col] ?? "", undefined, { numeric: true }));
  return dir === "asc" ? sorted : sorted.reverse();
}

export function CsvTable({ content, delimiter = "," }: { content: string; delimiter?: string }) {
  const t = useTranslations("files");
  const [sort, setSort] = useState<{ col: number; dir: "asc" | "desc" } | null>(null);
  const all = useMemo(() => parseCsv(content, delimiter), [content, delimiter]);
  const [header, ...body] = all;
  const rows = sort ? sortRows(body, sort.col, sort.dir) : body;
  if (!header) return null;
  return (
    <div className="jax-scroll h-full overflow-auto">
      <table className="w-full border-collapse text-[12px]">
        <thead className="sticky top-0 bg-surface-2">
          <tr>
            {header.map((h, i) => (
              <th
                key={i}
                className="cursor-pointer whitespace-nowrap border-b border-line px-2 py-1 text-left text-body-ink"
                onClick={() => setSort((s) => ({ col: i, dir: s?.col === i && s.dir === "asc" ? "desc" : "asc" }))}
              >
                {h}
                {sort?.col === i ? (sort.dir === "asc" ? " ↑" : " ↓") : ""}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((r, ri) => (
            <tr key={ri} className="border-b border-line/40">
              {r.map((c, ci) => <td key={ci} className="whitespace-nowrap px-2 py-1 text-body-ink">{c}</td>)}
            </tr>
          ))}
        </tbody>
      </table>
      {rows.length === 0 ? <p className="p-2 text-[12px] text-muted">{t("empty")}</p> : null}
    </div>
  );
}
