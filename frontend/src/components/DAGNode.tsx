import React from "react";
import { useCurrentFrame, spring, useVideoConfig, interpolate } from "remotion";
import { colors, fonts } from "../styles/theme";

interface Props {
  label: string;
  x: number;
  y: number;
  startFrame?: number;
  color?: string;
  status?: "pending" | "running" | "done" | "failed";
  size?: number;
}

const statusColors: Record<string, string> = {
  pending: colors.grayDark,
  running: colors.primary,
  done: colors.green,
  failed: colors.red,
};

export const DAGNode: React.FC<Props> = ({
  label,
  x,
  y,
  startFrame = 0,
  color,
  status = "pending",
  size = 48,
}) => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  const scale = spring({
    frame: frame - startFrame,
    fps,
    config: { damping: 12, mass: 0.5, stiffness: 200 },
  });

  const nodeColor = color || statusColors[status];
  const pulseOpacity =
    status === "running"
      ? 0.3 + 0.3 * Math.sin(frame * 0.15)
      : 0;

  return (
    <div
      style={{
        position: "absolute",
        left: x - size / 2,
        top: y - size / 2,
        width: size,
        height: size,
        borderRadius: 12,
        backgroundColor: `${nodeColor}22`,
        border: `2px solid ${nodeColor}`,
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        transform: `scale(${scale})`,
        boxShadow: `0 0 ${20 + pulseOpacity * 30}px ${nodeColor}${Math.round(pulseOpacity * 255).toString(16).padStart(2, "0")}`,
      }}
    >
      <span
        style={{
          fontSize: 13,
          color: colors.white,
          fontFamily: fonts.mono,
          fontWeight: 600,
          textAlign: "center",
          lineHeight: 1.2,
        }}
      >
        {label}
      </span>
    </div>
  );
};
