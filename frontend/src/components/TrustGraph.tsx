import React from "react";
import { useCurrentFrame, interpolate } from "remotion";
import { colors, fonts } from "../styles/theme";

interface DataPoint {
  label: string;
  value: number;
}

interface Props {
  data: DataPoint[];
  startFrame?: number;
  width?: number;
  height?: number;
  color?: string;
  showLabels?: boolean;
}

export const TrustGraph: React.FC<Props> = ({
  data,
  startFrame = 0,
  width = 500,
  height = 200,
  color = colors.secondary,
  showLabels = true,
}) => {
  const frame = useCurrentFrame();
  const padding = { top: 20, right: 20, bottom: 30, left: 50 };
  const chartW = width - padding.left - padding.right;
  const chartH = height - padding.top - padding.bottom;

  const minVal = Math.min(...data.map((d) => d.value)) - 0.05;
  const maxVal = Math.max(...data.map((d) => d.value)) + 0.05;

  const points = data.map((d, i) => ({
    x: padding.left + (i / (data.length - 1)) * chartW,
    y: padding.top + (1 - (d.value - minVal) / (maxVal - minVal)) * chartH,
    label: d.label,
    value: d.value,
  }));

  const drawProgress = interpolate(
    frame,
    [startFrame, startFrame + 45],
    [0, 1],
    { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
  );

  const pathData = points
    .map((p, i) => `${i === 0 ? "M" : "L"} ${p.x} ${p.y}`)
    .join(" ");

  // Calculate total path length approximation
  let totalLen = 0;
  for (let i = 1; i < points.length; i++) {
    const dx = points[i].x - points[i - 1].x;
    const dy = points[i].y - points[i - 1].y;
    totalLen += Math.sqrt(dx * dx + dy * dy);
  }

  return (
    <svg width={width} height={height}>
      {/* Grid lines */}
      {[0, 0.25, 0.5, 0.75, 1].map((frac) => {
        const y = padding.top + (1 - frac) * chartH;
        const val = minVal + frac * (maxVal - minVal);
        return (
          <g key={frac}>
            <line
              x1={padding.left}
              y1={y}
              x2={width - padding.right}
              y2={y}
              stroke={colors.grayDarker}
              strokeWidth={1}
            />
            <text
              x={padding.left - 8}
              y={y + 4}
              textAnchor="end"
              fill={colors.grayDark}
              fontSize={11}
              fontFamily={fonts.mono}
            >
              {val.toFixed(2)}
            </text>
          </g>
        );
      })}

      {/* Line */}
      <path
        d={pathData}
        fill="none"
        stroke={color}
        strokeWidth={3}
        strokeDasharray={totalLen}
        strokeDashoffset={totalLen * (1 - drawProgress)}
        strokeLinecap="round"
      />

      {/* Points */}
      {points.map((p, i) => {
        const pointProgress = interpolate(
          frame,
          [startFrame + (i / points.length) * 40, startFrame + (i / points.length) * 40 + 10],
          [0, 1],
          { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
        );
        return (
          <g key={i}>
            <circle
              cx={p.x}
              cy={p.y}
              r={5 * pointProgress}
              fill={color}
              opacity={pointProgress}
            />
            <circle
              cx={p.x}
              cy={p.y}
              r={8 * pointProgress}
              fill="none"
              stroke={color}
              strokeWidth={1}
              opacity={pointProgress * 0.4}
            />
            {showLabels && (
              <text
                x={p.x}
                y={height - 5}
                textAnchor="middle"
                fill={colors.gray}
                fontSize={11}
                fontFamily={fonts.sans}
                opacity={pointProgress}
              >
                {p.label}
              </text>
            )}
          </g>
        );
      })}
    </svg>
  );
};
