import type { ReactElement } from "react";
import { cardKey, projectSectionKey, type BoardEntry } from "@/lib/mission";
import { GroupFooter, GroupHeader } from "./GroupFrame";
import { ProjectCard } from "./ProjectCard";

export type BoardEntryOpts = {
  expandedDir: string | null;
  onToggleExpand: (key: string) => void;
  counts: Record<string, number> | null;
};

// One structure for plain AND group entries: wrapper(key=dir) > [header|null, cards container,
// footer|null] > ProjectCard(key=cardKey). A session card keeps the same element path when its
// project goes 1 -> 2 -> 1 sessions, so React never remounts it (focus, open Cancel dialog).
export function renderBoardEntry(entry: BoardEntry, { expandedDir, onToggleExpand, counts }: BoardEntryOpts): ReactElement {
  if (entry.kind === "plain") {
    const key = cardKey(entry.card);
    const expanded = key === expandedDir;
    return (
      <div key={entry.card.dir} data-card-dir={entry.card.dir} data-card-key={key} className={`min-w-0 grow basis-full sm:basis-[45%] ${expanded ? "lg:basis-[63%]" : "lg:basis-[30%]"}`}>
        {null}
        <div>
          {[<ProjectCard key={key} card={entry.card} issues={counts?.[entry.card.name.toLowerCase()]} expanded={expanded} onToggle={() => onToggleExpand(key)} />]}
        </div>
        {null}
      </div>
    );
  }
  const { project, cards } = entry;
  const sectionKey = projectSectionKey(project.dir);
  return (
    <div key={project.dir} data-card-key={sectionKey} data-card-dir={project.dir}
      className="flex min-w-0 w-full grow basis-full flex-col gap-3.5 rounded-lg border border-line bg-surface px-5 py-[18px]">
      <GroupHeader project={project} expanded={expandedDir === sectionKey} expandedKey={expandedDir}
        issues={counts?.[project.name.toLowerCase()]} onToggle={() => onToggleExpand(sectionKey)} />
      <div className="grid gap-3 sm:grid-cols-2">
        {cards.map((c, i) => {
          const key = cardKey(c);
          // Two cards per row; the expanded card and an odd last card take the whole row.
          const fullRow = key === expandedDir || (i === cards.length - 1 && cards.length % 2 === 1);
          return <ProjectCard key={key} card={c} expanded={key === expandedDir} onToggle={() => onToggleExpand(key)}
            rootAttrs={{ "data-card-key": key, className: fullRow ? "sm:col-span-full" : "" }} />;
        })}
      </div>
      <GroupFooter project={project} />
    </div>
  );
}
