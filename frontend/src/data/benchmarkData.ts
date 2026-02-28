export const overallStats = {
  delegato: { success: 83, total: 120, rate: 69.2 },
  naive: { success: 81, total: 120, rate: 67.5 },
  costMultiplier: 2.4,
  avgLatencyDelegato: 45.48,
  avgLatencyNaive: 15.68,
  totalTests: 310,
  coverage: 94,
  linesOfCode: 2416,
};

export const byDifficulty = [
  { label: "Easy", naive: 75.0, delegato: 75.0, naiveRaw: "27/36", delegatoRaw: "27/36" },
  { label: "Medium", naive: 66.7, delegato: 72.9, naiveRaw: "32/48", delegatoRaw: "35/48" },
  { label: "Hard", naive: 61.1, delegato: 58.3, naiveRaw: "22/36", delegatoRaw: "21/36" },
];

export const byCategory = [
  { label: "Research", naive: 70.0, delegato: 70.0 },
  { label: "Coding", naive: 50.0, delegato: 56.7 },
  { label: "Analysis", naive: 50.0, delegato: 50.0 },
  { label: "Writing", naive: 100.0, delegato: 100.0 },
];

export const trialProgression = [
  { trial: "Trial 1", delegato: 25.0, naive: 87.5 },
  { trial: "Trial 2", delegato: 87.5, naive: 75.0 },
  { trial: "Trial 3", delegato: 61.7, naive: 68.3 },
  { trial: "Trial 4", delegato: 69.2, naive: 67.5 },
];
