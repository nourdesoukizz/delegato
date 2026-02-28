import React from "react";
import { useCurrentFrame, spring, useVideoConfig, interpolate } from "remotion";
import { colors, fonts } from "../styles/theme";
import { DAGNode } from "../components/DAGNode";
import { DAGArrow } from "../components/DAGArrow";

const nodes = [
  { label: "Task", x: 960, y: 150, delay: 10 },
  { label: "Search", x: 500, y: 420, delay: 25 },
  { label: "Analyze", x: 960, y: 420, delay: 35 },
  { label: "Synthesize", x: 1420, y: 420, delay: 45 },
  { label: "Result", x: 960, y: 680, delay: 65 },
];

const arrows = [
  { from: 0, to: 1, delay: 20 },
  { from: 0, to: 2, delay: 30 },
  { from: 0, to: 3, delay: 40 },
  { from: 1, to: 4, delay: 55 },
  { from: 2, to: 4, delay: 58 },
  { from: 3, to: 4, delay: 61 },
];

export const Scene3a_Decomposition: React.FC = () => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  const titleOpacity = interpolate(frame, [0, 15], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });

  const stepOpacity = interpolate(frame, [5, 15], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });

  return (
    <div style={{ width: "100%", height: "100%", position: "relative" }}>
      {/* Step indicator */}
      <div
        style={{
          position: "absolute",
          top: 30,
          left: 60,
          opacity: stepOpacity,
          display: "flex",
          alignItems: "center",
          gap: 12,
        }}
      >
        <div
          style={{
            width: 36,
            height: 36,
            borderRadius: "50%",
            backgroundColor: colors.primary,
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            fontSize: 18,
            fontWeight: 700,
            color: colors.bg,
            fontFamily: fonts.sans,
          }}
        >
          1
        </div>
        <span style={{ fontSize: 28, fontWeight: 600, color: colors.white, fontFamily: fonts.sans }}>
          Task Decomposition
        </span>
      </div>

      {/* Subtitle */}
      <div
        style={{
          position: "absolute",
          top: 80,
          left: 108,
          fontSize: 18,
          color: colors.gray,
          fontFamily: fonts.sans,
          opacity: stepOpacity,
        }}
      >
        LLM breaks complex tasks into a DAG of subtasks
      </div>

      {/* Arrows */}
      {arrows.map((a, i) => (
        <DAGArrow
          key={i}
          x1={nodes[a.from].x}
          y1={nodes[a.from].y + 30}
          x2={nodes[a.to].x}
          y2={nodes[a.to].y - 30}
          startFrame={a.delay}
          color={colors.primary}
        />
      ))}

      {/* Nodes */}
      {nodes.map((n, i) => (
        <DAGNode
          key={i}
          label={n.label}
          x={n.x}
          y={n.y}
          startFrame={n.delay}
          color={i === 0 ? colors.primary : i === nodes.length - 1 ? colors.green : colors.secondary}
          size={i === 0 || i === nodes.length - 1 ? 56 : 64}
        />
      ))}
    </div>
  );
};
