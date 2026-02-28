import React from "react";
import { useCurrentFrame, spring, useVideoConfig } from "remotion";

interface Props {
  target: number;
  startFrame?: number;
  suffix?: string;
  prefix?: string;
  decimals?: number;
  fontSize?: number;
  color?: string;
  fontFamily?: string;
}

export const Counter: React.FC<Props> = ({
  target,
  startFrame = 0,
  suffix = "",
  prefix = "",
  decimals = 0,
  fontSize = 48,
  color = "#00D4FF",
  fontFamily = "Inter, sans-serif",
}) => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  const progress = spring({
    frame: frame - startFrame,
    fps,
    config: { damping: 30, mass: 1, stiffness: 80 },
  });

  const currentValue = target * progress;

  return (
    <span
      style={{
        fontSize,
        color,
        fontWeight: 700,
        fontFamily,
        fontVariantNumeric: "tabular-nums",
      }}
    >
      {prefix}{currentValue.toFixed(decimals)}{suffix}
    </span>
  );
};
