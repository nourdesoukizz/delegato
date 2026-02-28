export const colors = {
  bg: "#0A0E1A",
  bgLight: "#111827",
  bgCard: "#1A1F2E",
  primary: "#00D4FF",
  primaryDim: "#0099BB",
  secondary: "#A855F7",
  secondaryDim: "#7C3AED",
  green: "#22C55E",
  red: "#EF4444",
  orange: "#F59E0B",
  yellow: "#FACC15",
  white: "#F8FAFC",
  gray: "#94A3B8",
  grayDark: "#475569",
  grayDarker: "#1E293B",
} as const;

export const fonts = {
  mono: "JetBrains Mono, Fira Code, Consolas, monospace",
  sans: "Inter, SF Pro Display, -apple-system, sans-serif",
} as const;

export const springPresets = {
  snappy: { damping: 15, mass: 0.5, stiffness: 200 },
  gentle: { damping: 20, mass: 1, stiffness: 100 },
  bouncy: { damping: 10, mass: 0.8, stiffness: 180 },
  slow: { damping: 30, mass: 1.5, stiffness: 80 },
} as const;
