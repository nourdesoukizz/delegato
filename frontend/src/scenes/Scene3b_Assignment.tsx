import React from "react";
import { useCurrentFrame, spring, useVideoConfig, interpolate } from "remotion";
import { colors, fonts } from "../styles/theme";
import { AnimatedBar } from "../components/AnimatedBar";

const weights = [
  { label: "Capability Match", weight: 0.35, value: 92, color: colors.primary },
  { label: "Trust Score", weight: 0.30, value: 78, color: colors.secondary },
  { label: "Availability", weight: 0.20, value: 100, color: colors.green },
  { label: "Cost Efficiency", weight: 0.15, value: 85, color: colors.orange },
];

export const Scene3b_Assignment: React.FC = () => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  const stepOpacity = interpolate(frame, [0, 12], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });

  const formulaOpacity = interpolate(frame, [15, 30], [0, 1], {
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
          2
        </div>
        <span style={{ fontSize: 28, fontWeight: 600, color: colors.white, fontFamily: fonts.sans }}>
          Multi-Objective Assignment
        </span>
      </div>

      {/* Scoring formula */}
      <div
        style={{
          position: "absolute",
          top: 140,
          left: 0,
          width: "100%",
          textAlign: "center",
          opacity: formulaOpacity,
        }}
      >
        <span
          style={{
            fontFamily: fonts.mono,
            fontSize: 24,
            color: colors.white,
          }}
        >
          score = <span style={{ color: colors.primary }}>0.35</span> × cap +{" "}
          <span style={{ color: colors.secondary }}>0.30</span> × trust +{" "}
          <span style={{ color: colors.green }}>0.20</span> × avail +{" "}
          <span style={{ color: colors.orange }}>0.15</span> × cost
        </span>
      </div>

      {/* Animated bars */}
      <div
        style={{
          position: "absolute",
          top: 240,
          left: 0,
          width: "100%",
          display: "flex",
          justifyContent: "center",
        }}
      >
        <div style={{ width: 700 }}>
          {weights.map((w, i) => (
            <AnimatedBar
              key={i}
              label={`${w.label} (×${w.weight})`}
              value={w.value}
              color={w.color}
              startFrame={30 + i * 10}
              width={700}
              height={28}
            />
          ))}

          {/* Total score */}
          <div
            style={{
              marginTop: 40,
              textAlign: "center",
              opacity: interpolate(frame, [70, 80], [0, 1], {
                extrapolateLeft: "clamp",
                extrapolateRight: "clamp",
              }),
            }}
          >
            <span style={{ fontFamily: fonts.mono, fontSize: 20, color: colors.gray }}>
              Final Score:{" "}
            </span>
            <span
              style={{
                fontFamily: fonts.mono,
                fontSize: 36,
                fontWeight: 700,
                color: colors.primary,
                textShadow: `0 0 20px ${colors.primary}66`,
              }}
            >
              0.886
            </span>
            <span style={{ fontFamily: fonts.mono, fontSize: 20, color: colors.gray }}>
              {" → "}
            </span>
            <span
              style={{
                fontFamily: fonts.sans,
                fontSize: 24,
                fontWeight: 600,
                color: colors.green,
              }}
            >
              Assign to Searcher
            </span>
          </div>
        </div>
      </div>
    </div>
  );
};
