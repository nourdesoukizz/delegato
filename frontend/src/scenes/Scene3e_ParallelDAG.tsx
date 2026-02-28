import React from "react";
import { useCurrentFrame, spring, useVideoConfig, interpolate } from "remotion";
import { colors, fonts } from "../styles/theme";
import { DAGNode } from "../components/DAGNode";
import { DAGArrow } from "../components/DAGArrow";

type NodeStatus = "pending" | "running" | "done" | "failed";

const nodes = [
  { label: "Search", x: 300, y: 250 },
  { label: "Analyze", x: 700, y: 250 },
  { label: "Fact\nCheck", x: 700, y: 500 },
  { label: "Write", x: 1100, y: 375 },
  { label: "Verify", x: 1500, y: 375 },
];

const arrows = [
  { from: 0, to: 1 },
  { from: 1, to: 2 },
  { from: 1, to: 3 },
  { from: 2, to: 3 },
  { from: 3, to: 4 },
];

export const Scene3e_ParallelDAG: React.FC = () => {
  const frame = useCurrentFrame();

  // Animated statuses over time
  const getStatus = (nodeIdx: number): NodeStatus => {
    if (nodeIdx === 0) {
      if (frame < 15) return "pending";
      if (frame < 30) return "running";
      return "done";
    }
    if (nodeIdx === 1) {
      if (frame < 30) return "pending";
      if (frame < 45) return "running";
      return "done";
    }
    if (nodeIdx === 2) {
      if (frame < 35) return "pending";
      if (frame < 50) return "running";
      return "done";
    }
    if (nodeIdx === 3) {
      if (frame < 50) return "pending";
      if (frame < 60) return "running";
      if (frame < 65) return "failed";
      if (frame < 75) return "running"; // retry
      return "done";
    }
    if (nodeIdx === 4) {
      if (frame < 75) return "pending";
      if (frame < 85) return "running";
      return "done";
    }
    return "pending";
  };

  const stepOpacity = interpolate(frame, [0, 12], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });

  // Show retry label
  const retryOpacity = interpolate(frame, [62, 68], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });
  const retryFade = frame >= 75 ? interpolate(frame, [75, 80], [1, 0], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  }) : retryOpacity;

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
          5
        </div>
        <span style={{ fontSize: 28, fontWeight: 600, color: colors.white, fontFamily: fonts.sans }}>
          Parallel Execution with Retry
        </span>
      </div>

      {/* Arrows */}
      {arrows.map((a, i) => (
        <DAGArrow
          key={i}
          x1={nodes[a.from].x + 32}
          y1={nodes[a.from].y}
          x2={nodes[a.to].x - 32}
          y2={nodes[a.to].y}
          startFrame={10 + i * 5}
          color={colors.primaryDim}
        />
      ))}

      {/* Nodes */}
      {nodes.map((n, i) => (
        <DAGNode
          key={i}
          label={n.label}
          x={n.x}
          y={n.y}
          startFrame={5 + i * 5}
          status={getStatus(i)}
          size={60}
        />
      ))}

      {/* Retry indicator */}
      <div
        style={{
          position: "absolute",
          left: nodes[3].x - 40,
          top: nodes[3].y + 50,
          fontFamily: fonts.mono,
          fontSize: 16,
          color: colors.orange,
          fontWeight: 600,
          opacity: retryFade,
        }}
      >
        ↻ retry 1/2
      </div>

      {/* Progress bar */}
      <div
        style={{
          position: "absolute",
          bottom: 100,
          left: 200,
          right: 200,
        }}
      >
        <div
          style={{
            display: "flex",
            justifyContent: "space-between",
            fontFamily: fonts.mono,
            fontSize: 14,
            color: colors.gray,
            marginBottom: 8,
          }}
        >
          <span>Pipeline Progress</span>
          <span>
            {nodes.filter((_, i) => getStatus(i) === "done").length}/{nodes.length} complete
          </span>
        </div>
        <div
          style={{
            width: "100%",
            height: 8,
            backgroundColor: colors.grayDarker,
            borderRadius: 4,
            overflow: "hidden",
          }}
        >
          <div
            style={{
              width: `${(nodes.filter((_, i) => getStatus(i) === "done").length / nodes.length) * 100}%`,
              height: "100%",
              backgroundColor: colors.green,
              borderRadius: 4,
              transition: "width 0.3s ease",
            }}
          />
        </div>
      </div>
    </div>
  );
};
