"use client";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

export function Markdown({ children }: { children: string }) {
  return (
    <div className="jax-prose text-[13px] leading-[1.6] text-body-ink">
      <ReactMarkdown
        remarkPlugins={[remarkGfm]}
        components={{
          a: (p) => <a {...p} target="_blank" rel="noreferrer" className="text-brand hover:underline" />,
          code: (p) => <code {...p} className="rounded bg-surface-2 px-1 py-0.5 font-mono text-[12px] text-ink" />,
        }}
      >
        {children}
      </ReactMarkdown>
    </div>
  );
}
