import React from "react";
import { useCurrentFrame, spring, useVideoConfig, interpolate } from "remotion";

interface Props {
  value: number;
  maxValue?: number;
  label: string;
  color: string;
  startFrame?: number;
  width?: number;
  height?: number;
  showValue?: boolean;
  suffix?: string;
}

export const AnimatedBar: React.FC<Props> = ({
  value,
  maxValue = 100,
  label,
  color,
  startFrame = 0,
  width = 400,
  height = 32,
  showValue = true,
  suffix = "%",
}) => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  const progress = spring({
    frame: frame - startFrame,
    fps,
    config: { damping: 20, mass: 1, stiffness: 100 },
  });

  const barWidth = (value / maxValue) * width * progress;
  const displayValue = (value * progress).toFixed(1);

  const opacity = interpolate(frame, [startFrame, startFrame + 8], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });

  return (
    <div style={{ opacity, marginBottom: 12 }}>
      <div
        style={{
          display: "flex",
          justifyContent: "space-between",
          marginBottom: 6,
          fontFamily: "Inter, sans-serif",
          fontSize: 16,
          color: "#94A3B8",
        }}
      >
        <span>{label}</span>
        {showValue && (
          <span style={{ color, fontWeight: 600 }}>
            {displayValue}{suffix}
          </span>
        )}
      </div>
      <div
        style={{
          width,
          height,
          backgroundColor: "rgba(255,255,255,0.05)",
          borderRadius: height / 2,
          overflow: "hidden",
        }}
      >
        <div
          style={{
            width: barWidth,
            height: "100%",
            backgroundColor: color,
            borderRadius: height / 2,
            boxShadow: `0 0 20px ${color}66`,
          }}
        />
      </div>
    </div>
  );
};
