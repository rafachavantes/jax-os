"use client";

import { useState } from "react";
import type { MetricPoint } from "@/server/db/metrics"; // type-only: erased at build

// Normalizes points into a 0-100 x 0-40 viewBox polyline. Null values break the
// line (gap), matching "probe failed that round" honestly instead of drawing 0.
export function pathFrom(points: (number | null)[], min: number, max: number): string {
  const span = max - min || 1;
  const n = points.length;
  let d = "";
  let pen = false;
  points.forEach((v, i) => {
    if (v === null || Number.isNaN(v)) {
      pen = false;
      return;
    }
    const x = n === 1 ? 50 : (i / (n - 1)) * 100;
    const y = 40 - ((v - min) / span) * 36 - 2;
    d += `${pen ? "L" : "M"}${x.toFixed(2)},${y.toFixed(2)}`;
    pen = true;
  });
  return d;
}

export function Sparkline({ values }: { values: (number | null)[] }) {
  const nums = values.filter((v): v is number => v !== null);
  if (nums.length === 0) return <div className="h-10" />;
  const d = pathFrom(values, Math.min(...nums), Math.max(...nums));
  return (
    <svg viewBox="0 0 100 40" preserveAspectRatio="none" className="h-10 w-full text-muted" aria-hidden>
      <path d={d} fill="none" stroke="currentColor" strokeWidth={2} strokeLinejoin="round" strokeLinecap="round" vectorEffect="non-scaling-stroke" />
    </svg>
  );
}

type LineChartProps = {
  points: MetricPoint[];
  maxPoints?: MetricPoint[] | null; // 90d: the max series (lighter line, same hue)
  field: "cpuPct" | "memUsedMb";
  colorClass: string; // "text-brand" (CPU) | "text-accent" (Mem) — fixed identity
  format: (v: number) => string;
};

export function LineChart({ points, maxPoints, field, colorClass, format }: LineChartProps) {
  const [hover, setHover] = useState<number | null>(null);
  const values = points.map((p) => p[field]);
  const maxValues = maxPoints?.map((p) => p[field]) ?? [];
  const nums = [...values, ...maxValues].filter((v): v is number => v !== null);
  if (nums.length === 0) return <div className="h-36" />;
  const lo = Math.min(...nums, 0);
  const hi = Math.max(...nums);
  const hoverPoint = hover !== null ? points[hover] : null;

  return (
    <div className={`relative ${colorClass}`}>
      <svg
        viewBox="0 0 100 40"
        preserveAspectRatio="none"
        className="h-36 w-full"
        onMouseMove={(e) => {
          const r = e.currentTarget.getBoundingClientRect();
          // single point: length-1 division is 0/0 → NaN hover
          const i = points.length > 1 ? Math.round(((e.clientX - r.left) / r.width) * (points.length - 1)) : 0;
          setHover(Math.max(0, Math.min(points.length - 1, i)));
        }}
        onMouseLeave={() => setHover(null)}
      >
        <path d={`${pathFrom(values, lo, hi)}`} fill="none" stroke="currentColor" strokeWidth={2} strokeLinejoin="round" strokeLinecap="round" vectorEffect="non-scaling-stroke" />
        {maxPoints ? (
          <path d={pathFrom(maxValues, lo, hi)} fill="none" stroke="currentColor" strokeOpacity={0.35} strokeWidth={2} strokeLinejoin="round" strokeLinecap="round" vectorEffect="non-scaling-stroke" />
        ) : null}
        {hover !== null ? (
          <line x1={points.length > 1 ? (hover / (points.length - 1)) * 100 : 50} x2={points.length > 1 ? (hover / (points.length - 1)) * 100 : 50} y1={0} y2={40} stroke="currentColor" strokeOpacity={0.3} strokeWidth={1} vectorEffect="non-scaling-stroke" />
        ) : null}
      </svg>
      {/* axis labels + tooltip wear text tokens, never the series color */}
      <div className="pointer-events-none absolute left-0 top-0 text-[10px] text-muted">{format(hi)}</div>
      <div className="pointer-events-none absolute bottom-0 left-0 text-[10px] text-muted">{format(lo)}</div>
      {hoverPoint ? (
        <div className="pointer-events-none absolute right-0 top-0 rounded-md border border-line bg-surface-2 px-2 py-1 font-mono text-[11px] text-body-ink">
          {new Date(hoverPoint.ts).toLocaleTimeString()} · {hoverPoint[field] !== null ? format(hoverPoint[field] as number) : "—"}
        </div>
      ) : null}
    </div>
  );
}
