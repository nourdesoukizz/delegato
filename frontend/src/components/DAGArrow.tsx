import React from "react";
import { useCurrentFrame, interpolate } from "remotion";
import { colors } from "../styles/theme";

interface Props {
  x1: number;
  y1: number;
  x2: number;
  y2: number;
  startFrame?: number;
  duration?: number;
  color?: string;
}

export const DAGArrow: React.FC<Props> = ({
  x1,
  y1,
  x2,
  y2,
  startFrame = 0,
  duration = 15,
  color = colors.primary,
}) => {
  const frame = useCurrentFrame();

  const progress = interpolate(frame, [startFrame, startFrame + duration], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });

  const dx = x2 - x1;
  const dy = y2 - y1;
  const length = Math.sqrt(dx * dx + dy * dy);
  const angle = Math.atan2(dy, dx);

  // Arrow head
  const headSize = 8;
  const endX = x1 + dx * progress;
  const endY = y1 + dy * progress;

  return (
    <svg
      style={{ position: "absolute", inset: 0, pointerEvents: "none" }}
      width="1920"
      height="1080"
    >
      <line
        x1={x1}
        y1={y1}
        x2={endX}
        y2={endY}
        stroke={color}
        strokeWidth={2}
        opacity={0.6}
      />
      {progress > 0.1 && (
        <polygon
          points={`
            ${endX},${endY}
            ${endX - headSize * Math.cos(angle - 0.5)},${endY - headSize * Math.sin(angle - 0.5)}
            ${endX - headSize * Math.cos(angle + 0.5)},${endY - headSize * Math.sin(angle + 0.5)}
          `}
          fill={color}
          opacity={0.8}
        />
      )}
    </svg>
  );
};
