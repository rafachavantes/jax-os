"use client";

import { useEffect, useRef, useState } from "react";

export type MenuItem = { id: string; label: string; onSelect: () => void; danger?: boolean; hint?: string };
type MenuPos = { x: number; y: number };
export type MenuSelection = { id: string; x: number; y: number };

// One shared positioned menu, opened by right-click or a "···" trigger (§5 item 5) — no menu
// dependency: ~6 static items, a small local `role="menu"` div is enough (AGENTS.md "no shadcn
// until needed"). Tracks WHICH row opened it (round 3 F1) — a bare pos let every row's own
// `menu.pos ? <ContextMenu> : null` check pass at once, mounting one overlapping menu per row.
export function useContextMenu() {
  const [selection, setSelection] = useState<MenuSelection | null>(null);
  return {
    selection,
    open: (id: string, x: number, y: number) => setSelection({ id, x, y }),
    close: () => setSelection(null),
  };
}

export function ContextMenu({ pos, items, onClose }: { pos: MenuPos; items: MenuItem[]; onClose: () => void }) {
  const ref = useRef<HTMLDivElement>(null);
  useEffect(() => {
    const onDocDown = (e: MouseEvent) => {
      if (ref.current && !ref.current.contains(e.target as Node)) onClose();
    };
    const onKey = (e: KeyboardEvent) => { if (e.key === "Escape") onClose(); };
    document.addEventListener("mousedown", onDocDown);
    document.addEventListener("keydown", onKey);
    return () => {
      document.removeEventListener("mousedown", onDocDown);
      document.removeEventListener("keydown", onKey);
    };
  }, [onClose]);

  return (
    <div
      ref={ref}
      role="menu"
      style={{ position: "fixed", top: pos.y, left: pos.x }}
      className="z-50 min-w-[170px] rounded-md border border-line bg-surface-2 py-1 shadow-lg"
    >
      {items.map((item) => (
        <button
          key={item.id}
          type="button"
          role="menuitem"
          onClick={() => { item.onSelect(); onClose(); }}
          className={`block w-full px-3 py-1.5 text-left hover:bg-surface-3 ${
            item.danger ? "text-danger" : "text-body-ink"
          }`}
        >
          <span className="block text-[13px]">{item.label}</span>
          {item.hint ? <span className="block text-[11px] text-muted">{item.hint}</span> : null}
        </button>
      ))}
    </div>
  );
}
