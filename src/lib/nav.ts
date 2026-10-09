import {
  Blocks,
  Coins,
  FolderGit2,
  Kanban,
  LayoutDashboard,
  ScrollText,
  Server,
  SquareTerminal,
  type LucideIcon,
} from "lucide-react";

export type SidebarMode = "expanded" | "collapsed" | "hidden";

export type NavItem = {
  key: "mission" | "kanban" | "sessions" | "files" | "tokens" | "server" | "audit" | "tools";
  href: string;
  icon: LucideIcon;
  phase: number;
  badge?: string;
  // Decision 19: kept in NAV_ITEMS (navItemFor must keep resolving titles for these
  // routes) but filtered out of NavList/MobileNav rendering only (Task 9).
  hidden?: boolean;
};

export const NAV_ITEMS: NavItem[] = [
  { key: "mission", href: "/", icon: LayoutDashboard, phase: 2 },
  { key: "kanban", href: "/kanban", icon: Kanban, phase: 4 },
  { key: "sessions", href: "/tmux", icon: SquareTerminal, phase: 3 },
  { key: "files", href: "/files", icon: FolderGit2, phase: 5 },
  { key: "tokens", href: "/tokens", icon: Coins, phase: 6 },
  { key: "server", href: "/health", icon: Server, phase: 7, hidden: true },
  { key: "audit", href: "/audit", icon: ScrollText, phase: 8, hidden: true },
  { key: "tools", href: "/settings", icon: Blocks, phase: 9 },
];

export function navItemFor(pathname: string): NavItem {
  const match = NAV_ITEMS.filter(
    (i) => i.href !== "/" && (pathname === i.href || pathname.startsWith(i.href + "/")),
  ).sort((a, b) => b.href.length - a.href.length)[0];
  return match ?? NAV_ITEMS[0];
}
