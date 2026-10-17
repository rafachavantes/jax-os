import { TriangleAlert } from "lucide-react";

export function SourceWarning({ label, detail, aside }: { label: string; detail?: string; aside?: React.ReactNode }) {
  return (
    <div className="flex items-start gap-2.5 rounded-md border border-warning bg-warning-soft px-3.5 py-3">
      <TriangleAlert className="mt-0.5 h-4 w-4 flex-none text-warning" />
      <div className="flex min-w-0 flex-col gap-0.5">
        <span className="text-[12.5px] font-medium text-warning">{label}</span>
        {detail ? (
          <span className="break-all font-mono text-[11px] text-muted">{detail}</span>
        ) : null}
        {aside ? <span className="text-[11px] text-muted">{aside}</span> : null}
      </div>
    </div>
  );
}
