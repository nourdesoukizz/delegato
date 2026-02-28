import React from "react";
import { useCurrentFrame, interpolate, spring, useVideoConfig } from "remotion";
import { random } from "remotion";
import { colors, fonts } from "../styles/theme";

const AGENTS = [
  { id: 0, label: "A1", baseX: 400, baseY: 300 },
  { id: 1, label: "A2", baseX: 960, baseY: 200 },
  { id: 2, label: "A3", baseX: 1500, baseY: 350 },
  { id: 3, label: "A4", baseX: 600, baseY: 700 },
  { id: 4, label: "A5", baseX: 1300, baseY: 680 },
];

export const Scene2_TheProblem: React.FC = () => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  // Chaotic movement
  const agentPositions = AGENTS.map((a) => ({
    ...a,
    x: a.baseX + Math.sin(frame * 0.05 + a.id * 2) * 80 + Math.cos(frame * 0.03 + a.id) * 40,
    y: a.baseY + Math.cos(frame * 0.04 + a.id * 1.5) * 60 + Math.sin(frame * 0.06 + a.id) * 30,
  }));

  const titleOpacity = interpolate(frame, [0, 20], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });

  const subtitleOpacity = interpolate(frame, [50, 70], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });

  // Tangled lines between agents
  const connections: Array<[number, number]> = [
    [0, 1], [1, 2], [2, 3], [3, 4], [4, 0], [0, 2], [1, 3], [2, 4], [3, 0], [1, 4],
  ];

  return (
    <div
      style={{
        width: "100%",
        height: "100%",
        position: "relative",
      }}
    >
      {/* Tangled red lines */}
      <svg
        style={{ position: "absolute", inset: 0 }}
        width="1920"
        height="1080"
      >
        {connections.map(([from, to], i) => {
          const lineOpacity = interpolate(frame, [10 + i * 3, 20 + i * 3], [0, 0.4], {
            extrapolateLeft: "clamp",
            extrapolateRight: "clamp",
          });
          const a = agentPositions[from];
          const b = agentPositions[to];
          // Curved lines for tangled effect
          const midX = (a.x + b.x) / 2 + Math.sin(frame * 0.03 + i) * 50;
          const midY = (a.y + b.y) / 2 + Math.cos(frame * 0.04 + i) * 40;
          return (
            <path
              key={i}
              d={`M ${a.x} ${a.y} Q ${midX} ${midY} ${b.x} ${b.y}`}
              stroke={colors.red}
              strokeWidth={2}
              fill="none"
              opacity={lineOpacity}
            />
          );
        })}
      </svg>

      {/* Agent circles */}
      {agentPositions.map((a, i) => {
        const circleSpring = spring({
          frame: frame - i * 5,
          fps,
          config: { damping: 12, mass: 0.5, stiffness: 180 },
        });
        return (
          <div
            key={a.id}
            style={{
              position: "absolute",
              left: a.x - 35,
              top: a.y - 35,
              width: 70,
              height: 70,
              borderRadius: "50%",
              backgroundColor: `${colors.red}22`,
              border: `2px solid ${colors.red}88`,
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              transform: `scale(${circleSpring})`,
              boxShadow: `0 0 20px ${colors.red}33`,
            }}
          >
            <span style={{ color: colors.white, fontFamily: fonts.mono, fontSize: 20, fontWeight: 700 }}>
              {a.label}
            </span>
          </div>
        );
      })}

      {/* Title */}
      <div
        style={{
          position: "absolute",
          top: 80,
          width: "100%",
          textAlign: "center",
          opacity: titleOpacity,
        }}
      >
        <span
          style={{
            fontSize: 52,
            fontWeight: 700,
            color: colors.white,
            fontFamily: fonts.sans,
          }}
        >
          The Problem
        </span>
      </div>

      {/* Subtitle */}
      <div
        style={{
          position: "absolute",
          bottom: 100,
          width: "100%",
          textAlign: "center",
          opacity: subtitleOpacity,
        }}
      >
        <span
          style={{
            fontSize: 30,
            color: colors.gray,
            fontFamily: fonts.sans,
            fontStyle: "italic",
          }}
        >
          Multi-agent systems lack organizational intelligence
        </span>
      </div>
    </div>
  );
};
