import type { DragEvent, ReactElement } from "react";
import { cardKey, projectSectionKey, type BoardEntry } from "@/lib/mission";
import { DEFAULT_WIDTH, type LayoutEntry, type ProjectWidth } from "@/lib/cockpitLayout";
import { GroupFooter, GroupHeader } from "./GroupFrame";
import { ProjectCard } from "./ProjectCard";
import { ProjectOrganizeHandle, readDraggedProject } from "./ProjectLayoutControls";

export type BoardEntryOrganize = {
  draft: LayoutEntry[];
  // One explicit pending-write flag: handle/select/buttons and drop handlers all read it.
  disabled: boolean;
  onMove: (dir: string, delta: -1 | 1) => void;
  onWidth: (dir: string, width: ProjectWidth) => void;
  onDrop: (dragged: string, target: string) => void;
};

export type BoardEntryOpts = {
  expandedDir: string | null;
  onToggleExpand: (key: string) => void;
  counts: Record<string, number> | null;
  widths?: Map<string, ProjectWidth>;
  organize?: BoardEntryOrganize;
};

// One structure for plain AND group entries: wrapper(key=dir) > [header|null, cards container,
// footer|null] > ProjectCard(key=cardKey). A session card keeps the same element path when its
// project goes 1 -> 2 -> 1 sessions, so React never remounts it (focus, open Cancel dialog).
// M2: the wrapper is one 12-column grid span (width preset, never expansion-dependent) and drops
// only that project's own saved order; drop handlers exist only while Organize is open.
export function renderBoardEntry(entry: BoardEntry, opts: BoardEntryOpts): ReactElement {
  const { organize } = opts;
  const dir = entry.kind === "plain" ? entry.card.dir : entry.project.dir;
  const width = opts.widths?.get(dir) ?? DEFAULT_WIDTH;
  const index = organize ? organize.draft.findIndex((e) => e.project === dir) : -1;
  const dropProps = organize
    ? {
        onDragOver: (e: DragEvent) => {
          e.preventDefault();
          e.dataTransfer.dropEffect = organize.disabled ? "none" : "move";
        },
        onDrop: (e: DragEvent) => {
          e.preventDefault();
          if (organize.disabled) return;
          const dragged = readDraggedProject(e.dataTransfer, organize.draft);
          if (dragged) organize.onDrop(dragged, dir);
        },
      }
    : {};
  const handle = (name: string) =>
    organize ? (
      <ProjectOrganizeHandle
        dir={dir} name={name} width={width}
        index={index < 0 ? 0 : index} count={organize.draft.length}
        onMove={organize.onMove} onWidth={organize.onWidth}
        disabled={organize.disabled}
      />
    ) : null;

  if (entry.kind === "plain") {
    const key = cardKey(entry.card);
    const expanded = key === opts.expandedDir;
    return (
      <div key={entry.card.dir} data-card-dir={entry.card.dir} data-card-key={key} data-width={width}
        className="project-span" {...dropProps}>
        {handle(entry.card.name)}
        <div>
          {[<ProjectCard key={key} card={entry.card} issues={opts.counts?.[entry.card.name.toLowerCase()]} expanded={expanded} onToggle={() => opts.onToggleExpand(key)} />]}
        </div>
        {null}
      </div>
    );
  }
  const { project, cards } = entry;
  const sectionKey = projectSectionKey(project.dir);
  return (
    <div key={project.dir} data-card-key={sectionKey} data-card-dir={project.dir} data-width={width}
      className="project-span group-card flex min-w-0 flex-col gap-3.5 rounded-lg border border-line bg-surface px-5 py-[18px]" {...dropProps}>
      <div className="contents">
        {handle(project.name)}
        <GroupHeader project={project} expanded={opts.expandedDir === sectionKey} expandedKey={opts.expandedDir}
          issues={opts.counts?.[project.name.toLowerCase()]} onToggle={() => opts.onToggleExpand(sectionKey)} />
      </div>
      <div className="group-sessions">
        {cards.map((c, i) => {
          const key = cardKey(c);
          // The expanded card and an odd last card take the whole row; the project width itself
          // is never changed by expansion (spec "Saved project layout").
          const fullRow = key === opts.expandedDir || (i === cards.length - 1 && cards.length % 2 === 1);
          return <ProjectCard key={key} card={c} expanded={key === opts.expandedDir} onToggle={() => opts.onToggleExpand(key)}
            rootAttrs={{ "data-card-key": key, className: fullRow ? "group-session-full" : "" }} />;
        })}
      </div>
      <GroupFooter project={project} />
    </div>
  );
}
