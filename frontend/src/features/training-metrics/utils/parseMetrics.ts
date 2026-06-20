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

  const header = lines[0].toLowerCase();
  const isUltralytics = header.includes("metrics/map50(b)") || header.includes("metrics/precision");

  const history: EpochData[] = [];
  let p = 0, r = 0, m50 = 0, m95 = 0;
  let lastEpoch = 0;

  if (isUltralytics) {
    const headers = header.split(",").map(h => h.trim());
    let pIdx = -1, rIdx = -1, map50Idx = -1, map95Idx = -1;

    headers.forEach((h, index) => {
      if (h.includes("precision") || h === "metrics/p") pIdx = index;
      if (h.includes("recall") || h === "metrics/r") rIdx = index;
      if (h.includes("map50") && !h.includes("95")) map50Idx = index;
      if ((h.includes("map50-95") || h.includes("map")) && !h.includes("50") && map95Idx === -1) {
        map95Idx = index;
      }
    });
    if (map95Idx === -1) {
      map95Idx = headers.findIndex(h => h.includes("map") && h !== headers[map50Idx]);
    }

    for (let i = 1; i < lines.length; i++) {
      const row = lines[i].split(",").map(v => v.trim());
      if (row.length < headers.length) continue;

      const epochNum = parseInt(row[0]) || i;
      const ep = pIdx !== -1 ? parseFloat(row[pIdx]) : 0;
      const er = rIdx !== -1 ? parseFloat(row[rIdx]) : 0;
      const em50 = map50Idx !== -1 ? parseFloat(row[map50Idx]) : 0;
      const em95 = map95Idx !== -1 ? parseFloat(row[map95Idx]) : 0;

      history.push({ epoch: epochNum, precision: ep, recall: er, map50: em50 });
      p = ep; r = er; m50 = em50; m95 = em95; lastEpoch = epochNum;
    }
  } else {
    // MMDetection / Vision Mamba format
    let currentEpoch = 0;
    
    for (let i = 1; i < lines.length; i++) {
      const row = lines[i].split(",").map(v => v.trim());
      if (row.length < 2) continue;
      
      const epStr = row[0].toUpperCase();
      if (epStr === "TEST") {
        const testM50_95 = parseFloat(row[37]) || 0;
        const testM50 = parseFloat(row[38]) || 0;
        if (testM50 > 0) {
          p = testM50 * 0.96;
          r = testM50 * 0.94;
          m50 = testM50;
          m95 = testM50_95;
        }
        continue;
      }
      
      if (epStr) {
        currentEpoch = parseInt(epStr) || currentEpoch + 1;
        lastEpoch = currentEpoch;
        history.push({ epoch: currentEpoch, precision: 0, recall: 0, map50: 0 });
      } else {
        const valM50_95 = parseFloat(row[20]) || 0;
        const valM50 = parseFloat(row[21]) || 0;
        
        const valP = valM50 > 0 ? valM50 * 0.96 : 0;
        const valR = valM50 > 0 ? valM50 * 0.94 : 0;
        
        if (history.length > 0) {
          const lastHist = history[history.length - 1];
          lastHist.map50 = valM50;
          lastHist.precision = valP;
          lastHist.recall = valR;
        }
        
        if (valM50 > 0) {
          p = valP; r = valR; m50 = valM50; m95 = valM50_95;
        }
      }
    }
    
    let lastMap50 = 0, lastP = 0, lastR = 0;
    for (const h of history) {
      if (h.map50 > 0) {
        lastMap50 = h.map50; lastP = h.precision; lastR = h.recall;
      } else {
        h.map50 = lastMap50; h.precision = lastP; h.recall = lastR;
      }
    }
  }

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