export interface EpochData {
  epoch: number;
  precision: number;
  recall: number;
  map50: number;
}

export interface ParsedMetrics {
  precision: string;
  recall: string;
  map50: string;
  map50_95: string;
  f1: string;
  history: EpochData[];
}

export function parseMetrics(csvData: string): ParsedMetrics {
  if (!csvData) {
    return { precision: "N/A", recall: "N/A", map50: "N/A", map50_95: "N/A", f1: "N/A", history: [] };
  }

  const lines = csvData.trim().split("\n");
  if (lines.length < 2) {
    return { precision: "N/A", recall: "N/A", map50: "N/A", map50_95: "N/A", f1: "N/A", history: [] };
  }

  const headers = lines[0].split(",").map(h => h.trim().toLowerCase());
  
  // Find column indices
  let pIdx = -1, rIdx = -1, map50Idx = -1, map95Idx = -1;

  headers.forEach((header, index) => {
    if (header.includes("precision") || header === "metrics/p") pIdx = index;
    if (header.includes("recall") || header === "metrics/r") rIdx = index;
    // Map50 is usually metrics/mAP50(B)
    if (header.includes("map50") && !header.includes("95")) map50Idx = index;
    // Map50-95
    if (header.includes("map50-95") || header.includes("map") && !header.includes("50") && map95Idx === -1) {
      map95Idx = index;
    }
  });

  // Fallback map50-95 if strictly named
  if (map95Idx === -1) {
    map95Idx = headers.findIndex(h => h.includes("map") && h !== headers[map50Idx]);
  }

  // Parse all epochs for history
  const history: EpochData[] = [];
  for (let i = 1; i < lines.length; i++) {
    const row = lines[i].split(",").map(v => v.trim());
    if (row.length < headers.length) continue; // Skip malformed/empty lines

    const epochNum = parseInt(row[0]) || i;
    const ep = pIdx !== -1 ? parseFloat(row[pIdx]) : 0;
    const er = rIdx !== -1 ? parseFloat(row[rIdx]) : 0;
    const em50 = map50Idx !== -1 ? parseFloat(row[map50Idx]) : 0;

    history.push({
      epoch: epochNum,
      precision: ep,
      recall: er,
      map50: em50
    });
  }

  // Get the last valid row
  let lastRow = lines[lines.length - 1].split(",").map(v => v.trim());
  // If the last row is empty or malformed due to trailing newline, get the previous one
  if (lastRow.length < headers.length && lines.length > 2) {
      lastRow = lines[lines.length - 2].split(",").map(v => v.trim());
  }

  const p = pIdx !== -1 ? parseFloat(lastRow[pIdx]) : 0;
  const r = rIdx !== -1 ? parseFloat(lastRow[rIdx]) : 0;
  const m50 = map50Idx !== -1 ? parseFloat(lastRow[map50Idx]) : 0;
  const m95 = map95Idx !== -1 ? parseFloat(lastRow[map95Idx]) : 0;

  // F1 = 2 * (P * R) / (P + R)
  let f1 = 0;
  if (p + r > 0) {
    f1 = 2 * (p * r) / (p + r);
  }

  return {
    precision: p > 0 ? (p * 100).toFixed(1) + "%" : "N/A",
    recall: r > 0 ? (r * 100).toFixed(1) + "%" : "N/A",
    map50: m50 > 0 ? (m50 * 100).toFixed(1) + "%" : "N/A",
    map50_95: m95 > 0 ? (m95 * 100).toFixed(1) + "%" : "N/A",
    f1: f1 > 0 ? f1.toFixed(3) : "N/A",
    history
  };
}
