export type BreadcrumbSegment = { label: string; rel: string; isDir: boolean; collapsed: boolean };

// Extracted from FileEditor.tsx's inline breadcrumb JSX (spec §5). Every non-final segment is a
// directory by construction (an ancestor always is); the caller decides the final one via
// `finalIsDir`. More than 3 segments collapse the middle into one "…" placeholder spanning every
// hidden segment (FI-3), keeping the root + the last two segments legible. An empty `rel` (nothing
// selected) returns no segments at all — the caller renders just its own root crumb.
export function breadcrumbSegments(rel: string, finalIsDir: boolean): BreadcrumbSegment[] {
  if (rel === "") return [];
  const parts = rel.split("/");
  const mk = (i: number): BreadcrumbSegment => {
    const segRel = parts.slice(0, i + 1).join("/");
    const isLast = i === parts.length - 1;
    return { label: parts[i], rel: segRel, isDir: isLast ? finalIsDir : true, collapsed: false };
  };
  if (parts.length <= 3) return parts.map((_, i) => mk(i));
  const collapsedRel = parts.slice(0, parts.length - 2).join("/");
  return [
    { label: "…", rel: collapsedRel, isDir: true, collapsed: true },
    mk(parts.length - 2),
    mk(parts.length - 1),
  ];
}
