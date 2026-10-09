// Minimal in-repo LCS line diff for the Jax Rules diff view (spec §6.3: "NO
// new dependency"). O(n*m) DP table — fine for rule files (tens of lines,
// never large). Zero Node imports; safe for client bundling.

export type DiffLine = { type: "same" | "add" | "remove"; text: string };

export function diffLines(a: string, b: string): DiffLine[] {
  const linesA = a === "" ? [] : a.split("\n");
  const linesB = b === "" ? [] : b.split("\n");
  const n = linesA.length;
  const m = linesB.length;
  const lcs: number[][] = Array.from({ length: n + 1 }, () => new Array(m + 1).fill(0));
  for (let i = n - 1; i >= 0; i--) {
    for (let j = m - 1; j >= 0; j--) {
      lcs[i][j] = linesA[i] === linesB[j] ? lcs[i + 1][j + 1] + 1 : Math.max(lcs[i + 1][j], lcs[i][j + 1]);
    }
  }
  const out: DiffLine[] = [];
  let i = 0;
  let j = 0;
  while (i < n && j < m) {
    if (linesA[i] === linesB[j]) {
      out.push({ type: "same", text: linesA[i] });
      i++;
      j++;
    } else if (lcs[i + 1][j] >= lcs[i][j + 1]) {
      out.push({ type: "remove", text: linesA[i] });
      i++;
    } else {
      out.push({ type: "add", text: linesB[j] });
      j++;
    }
  }
  while (i < n) {
    out.push({ type: "remove", text: linesA[i] });
    i++;
  }
  while (j < m) {
    out.push({ type: "add", text: linesB[j] });
    j++;
  }
  return out;
}
