import React from "react";
import { useCurrentFrame, interpolate, spring, useVideoConfig } from "remotion";
import { colors, fonts } from "../styles/theme";
import { TrustGraph } from "../components/TrustGraph";
import { GlowBox } from "../components/GlowBox";

const trustData = [
  { label: "T0", value: 0.5 },
  { label: "T1", value: 0.55 },
  { label: "T2", value: 0.62 },
  { label: "T3", value: 0.58 },
  { label: "T4", value: 0.65 },
  { label: "T5", value: 0.71 },
  { label: "T6", value: 0.68 },
  { label: "T7", value: 0.75 },
];

export const Scene3d_TrustSafety: React.FC = () => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  const stepOpacity = interpolate(frame, [0, 12], [0, 1], {
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
          4
        </div>
        <span style={{ fontSize: 28, fontWeight: 600, color: colors.white, fontFamily: fonts.sans }}>
          Trust & Safety
        </span>
      </div>

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
        Time-decayed trust tracking with circuit breaker protection
      </div>

      {/* Trust graph */}
      <div
        style={{
          position: "absolute",
          top: 150,
          left: 100,
        }}
      >
        <div style={{ marginBottom: 16 }}>
          <span style={{ fontFamily: fonts.mono, fontSize: 16, color: colors.secondary }}>
            searcher.web_search trust score
          </span>
        </div>
        <TrustGraph
          data={trustData}
          startFrame={10}
          width={750}
          height={320}
          color={colors.secondary}
        />
      </div>

      {/* Circuit breaker */}
      <div
        style={{
          position: "absolute",
          top: 160,
          right: 100,
          width: 400,
        }}
      >
        <GlowBox color={colors.red} padding="28px">
          <div
            style={{
              fontFamily: fonts.mono,
              fontSize: 20,
              fontWeight: 700,
              color: colors.red,
              marginBottom: 16,
              opacity: interpolate(frame, [40, 50], [0, 1], {
                extrapolateLeft: "clamp",
                extrapolateRight: "clamp",
              }),
            }}
          >
            Circuit Breaker
          </div>
          <div
            style={{
              fontFamily: fonts.sans,
              fontSize: 16,
              color: colors.gray,
              lineHeight: 1.8,
              opacity: interpolate(frame, [50, 60], [0, 1], {
                extrapolateLeft: "clamp",
                extrapolateRight: "clamp",
              }),
            }}
          >
            Trust {"<"} 0.3 → agent suspended
          </div>
          <div
            style={{
              fontFamily: fonts.sans,
              fontSize: 16,
              color: colors.gray,
              lineHeight: 1.8,
              opacity: interpolate(frame, [55, 65], [0, 1], {
                extrapolateLeft: "clamp",
                extrapolateRight: "clamp",
              }),
            }}
          >
            3+ consecutive failures → blocked
          </div>
          <div
            style={{
              fontFamily: fonts.sans,
              fontSize: 16,
              color: colors.gray,
              lineHeight: 1.8,
              opacity: interpolate(frame, [60, 70], [0, 1], {
                extrapolateLeft: "clamp",
                extrapolateRight: "clamp",
              }),
            }}
          >
            Asymmetric updates: fail penalizes more
          </div>
        </GlowBox>

        {/* Features */}
        <div style={{ marginTop: 24 }}>
          <GlowBox color={colors.primary} padding="20px">
            <div
              style={{
                fontFamily: fonts.mono,
                fontSize: 16,
                color: colors.primary,
                opacity: interpolate(frame, [65, 75], [0, 1], {
                  extrapolateLeft: "clamp",
                  extrapolateRight: "clamp",
                }),
              }}
            >
              Time-based decay: older results fade
            </div>
          </GlowBox>
        </div>
      </div>
    </div>
  );
};
