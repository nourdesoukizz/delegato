import React from "react";
import { useCurrentFrame, spring, useVideoConfig, interpolate } from "remotion";
import { colors, fonts } from "../styles/theme";
import { GlowBox } from "../components/GlowBox";

const methods = [
  { name: "LLM_JUDGE", desc: "AI evaluates output quality", icon: "🧠", color: colors.secondary },
  { name: "REGEX", desc: "Pattern matching validation", icon: "⚡", color: colors.primary },
  { name: "SCHEMA", desc: "JSON schema conformance", icon: "📋", color: colors.orange },
  { name: "FUNCTION", desc: "Custom validation logic", icon: "⚙️", color: colors.green },
  { name: "NONE", desc: "Trust-based pass-through", icon: "✓", color: colors.gray },
];

export const Scene3c_Verification: React.FC = () => {
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
          3
        </div>
        <span style={{ fontSize: 28, fontWeight: 600, color: colors.white, fontFamily: fonts.sans }}>
          Multi-Method Verification
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
        5 verification methods with multi-judge consensus
      </div>

      {/* Verification method cards */}
      <div
        style={{
          position: "absolute",
          top: 160,
          left: 0,
          width: "100%",
          display: "flex",
          justifyContent: "center",
          gap: 24,
          padding: "0 60px",
        }}
      >
        {methods.map((m, i) => {
          const cardSpring = spring({
            frame: frame - 15 - i * 8,
            fps,
            config: { damping: 15, mass: 0.5, stiffness: 200 },
          });
          const cardOpacity = interpolate(
            frame,
            [15 + i * 8, 25 + i * 8],
            [0, 1],
            { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
          );

          return (
            <div
              key={i}
              style={{
                opacity: cardOpacity,
                transform: `translateY(${(1 - cardSpring) * 30}px) scale(${cardSpring})`,
                width: 300,
              }}
            >
              <GlowBox color={m.color} padding="24px">
                <div style={{ textAlign: "center" }}>
                  <div style={{ fontSize: 36, marginBottom: 12 }}>{m.icon}</div>
                  <div
                    style={{
                      fontSize: 20,
                      fontWeight: 700,
                      color: m.color,
                      fontFamily: fonts.mono,
                      marginBottom: 8,
                    }}
                  >
                    {m.name}
                  </div>
                  <div style={{ fontSize: 14, color: colors.gray, fontFamily: fonts.sans }}>
                    {m.desc}
                  </div>
                </div>
              </GlowBox>
            </div>
          );
        })}
      </div>

      {/* Judge voting visualization */}
      <div
        style={{
          position: "absolute",
          bottom: 120,
          width: "100%",
          display: "flex",
          justifyContent: "center",
          gap: 40,
          alignItems: "center",
          opacity: interpolate(frame, [65, 78], [0, 1], {
            extrapolateLeft: "clamp",
            extrapolateRight: "clamp",
          }),
        }}
      >
        {[1, 2, 3].map((judge, i) => {
          const vote = i < 2;
          const voteSpring = spring({
            frame: frame - 70 - i * 5,
            fps,
            config: { damping: 12, mass: 0.5, stiffness: 200 },
          });
          return (
            <div key={i} style={{ textAlign: "center", transform: `scale(${voteSpring})` }}>
              <div
                style={{
                  width: 60,
                  height: 60,
                  borderRadius: "50%",
                  backgroundColor: vote ? `${colors.green}22` : `${colors.red}22`,
                  border: `2px solid ${vote ? colors.green : colors.red}`,
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "center",
                  fontSize: 28,
                  marginBottom: 8,
                }}
              >
                {vote ? "✓" : "✗"}
              </div>
              <span style={{ fontFamily: fonts.mono, fontSize: 14, color: colors.gray }}>
                Judge {judge}
              </span>
            </div>
          );
        })}
        <div
          style={{
            fontSize: 20,
            fontFamily: fonts.mono,
            color: colors.green,
            fontWeight: 700,
          }}
        >
          2/3 = PASS
        </div>
      </div>
    </div>
  );
};
