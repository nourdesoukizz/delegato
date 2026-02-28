import React from "react";
import { useCurrentFrame, spring, useVideoConfig, interpolate } from "remotion";
import { colors, fonts } from "../styles/theme";
import { GlowBox } from "../components/GlowBox";

const components = [
  { name: "Task", subtitle: "Goal + constraints", y: 80, color: colors.primary },
  { name: "Decomposer", subtitle: "LLM-powered DAG builder", y: 240, color: colors.secondary },
  { name: "Assignment Scorer", subtitle: "Multi-objective ranking", y: 400, color: colors.orange },
  { name: "Agent Pool", subtitle: "Capability-based routing", y: 560, color: colors.green },
  { name: "Verifier", subtitle: "5-method validation", y: 720, color: colors.red },
  { name: "Trust Tracker", subtitle: "Bayesian adaptation", y: 880, color: colors.secondary },
];

export const Scene5_Architecture: React.FC = () => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  const titleOpacity = interpolate(frame, [0, 15], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });

  return (
    <div style={{ width: "100%", height: "100%", position: "relative" }}>
      {/* Title */}
      <div
        style={{
          position: "absolute",
          top: 20,
          width: "100%",
          textAlign: "center",
          opacity: titleOpacity,
        }}
      >
        <span style={{ fontSize: 36, fontWeight: 700, color: colors.white, fontFamily: fonts.sans }}>
          Architecture
        </span>
      </div>

      {/* Flow arrows + components */}
      <svg
        style={{ position: "absolute", inset: 0, pointerEvents: "none" }}
        width="1920"
        height="1080"
      >
        {components.slice(0, -1).map((comp, i) => {
          const drawProgress = interpolate(
            frame,
            [20 + i * 15, 35 + i * 15],
            [0, 1],
            { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
          );
          const nextComp = components[i + 1];
          return (
            <g key={i}>
              <line
                x1={960}
                y1={comp.y + 60}
                x2={960}
                y2={comp.y + 60 + (nextComp.y - comp.y - 60) * drawProgress}
                stroke={colors.primary}
                strokeWidth={2}
                opacity={0.5}
                strokeDasharray="6,4"
              />
              {drawProgress > 0.5 && (
                <circle
                  cx={960}
                  cy={comp.y + 60 + (nextComp.y - comp.y - 60) * Math.min(drawProgress, 0.95)}
                  r={4}
                  fill={colors.primary}
                  opacity={0.8}
                >
                  <animate
                    attributeName="r"
                    values="3;5;3"
                    dur="0.8s"
                    repeatCount="indefinite"
                  />
                </circle>
              )}
            </g>
          );
        })}
      </svg>

      {/* Component cards */}
      {components.map((comp, i) => {
        const cardSpring = spring({
          frame: frame - 10 - i * 12,
          fps,
          config: { damping: 15, mass: 0.5, stiffness: 200 },
        });
        const cardOpacity = interpolate(
          frame,
          [10 + i * 12, 20 + i * 12],
          [0, 1],
          { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
        );

        return (
          <div
            key={i}
            style={{
              position: "absolute",
              top: comp.y,
              left: "50%",
              transform: `translateX(-50%) translateY(${(1 - cardSpring) * 20}px) scale(${0.8 + cardSpring * 0.2})`,
              opacity: cardOpacity,
              width: 500,
            }}
          >
            <GlowBox color={comp.color} padding="14px 28px" borderRadius={12}>
              <div style={{ display: "flex", alignItems: "center", gap: 16 }}>
                <div
                  style={{
                    width: 8,
                    height: 8,
                    borderRadius: "50%",
                    backgroundColor: comp.color,
                    boxShadow: `0 0 12px ${comp.color}`,
                  }}
                />
                <div>
                  <div style={{ fontSize: 20, fontWeight: 700, color: comp.color, fontFamily: fonts.mono }}>
                    {comp.name}
                  </div>
                  <div style={{ fontSize: 14, color: colors.gray, fontFamily: fonts.sans }}>
                    {comp.subtitle}
                  </div>
                </div>
              </div>
            </GlowBox>
          </div>
        );
      })}
    </div>
  );
};
