import React from "react";
import { useCurrentFrame, spring, useVideoConfig, interpolate } from "remotion";
import { colors, fonts } from "../styles/theme";

export const Scene1_LogoReveal: React.FC = () => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  const letters = "delegato".split("");

  const tagline = "Intelligent Delegation for Multi-Agent Systems";
  const taglineOpacity = interpolate(frame, [60, 80], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });
  const taglineY = interpolate(frame, [60, 80], [20, 0], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });

  const stats = [
    { label: "310 tests", delay: 85 },
    { label: "94% coverage", delay: 95 },
    { label: "2,416 LOC", delay: 105 },
  ];

  return (
    <div
      style={{
        width: "100%",
        height: "100%",
        display: "flex",
        flexDirection: "column",
        alignItems: "center",
        justifyContent: "center",
        position: "relative",
      }}
    >
      {/* Logo */}
      <div style={{ display: "flex", marginBottom: 30 }}>
        {letters.map((letter, i) => {
          const letterSpring = spring({
            frame: frame - i * 5,
            fps,
            config: { damping: 10, mass: 0.8, stiffness: 180 },
          });
          const opacity = interpolate(
            frame,
            [i * 5, i * 5 + 8],
            [0, 1],
            { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
          );

          return (
            <span
              key={i}
              style={{
                fontSize: 120,
                fontWeight: 700,
                fontFamily: fonts.sans,
                color: i < 4 ? colors.primary : colors.secondary,
                opacity,
                transform: `translateY(${(1 - letterSpring) * -40}px) scale(${0.5 + letterSpring * 0.5})`,
                display: "inline-block",
                textShadow: `0 0 40px ${i < 4 ? colors.primary : colors.secondary}66`,
              }}
            >
              {letter}
            </span>
          );
        })}
      </div>

      {/* Tagline */}
      <div
        style={{
          fontSize: 28,
          color: colors.gray,
          fontFamily: fonts.sans,
          opacity: taglineOpacity,
          transform: `translateY(${taglineY}px)`,
          marginBottom: 40,
        }}
      >
        {tagline}
      </div>

      {/* Stat badges */}
      <div style={{ display: "flex", gap: 24 }}>
        {stats.map((stat, i) => {
          const badgeSpring = spring({
            frame: frame - stat.delay,
            fps,
            config: { damping: 15, mass: 0.5, stiffness: 200 },
          });
          const badgeOpacity = interpolate(
            frame,
            [stat.delay, stat.delay + 10],
            [0, 1],
            { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
          );

          return (
            <div
              key={i}
              style={{
                opacity: badgeOpacity,
                transform: `scale(${badgeSpring})`,
                padding: "10px 24px",
                borderRadius: 8,
                backgroundColor: `${colors.primary}15`,
                border: `1px solid ${colors.primary}33`,
                color: colors.primary,
                fontFamily: fonts.mono,
                fontSize: 18,
                fontWeight: 600,
              }}
            >
              {stat.label}
            </div>
          );
        })}
      </div>

      {/* DeepMind reference */}
      <div
        style={{
          position: "absolute",
          bottom: 60,
          fontSize: 16,
          color: colors.grayDark,
          fontFamily: fonts.mono,
          opacity: interpolate(frame, [110, 130], [0, 0.6], {
            extrapolateLeft: "clamp",
            extrapolateRight: "clamp",
          }),
        }}
      >
        Based on DeepMind's arXiv:2602.11865
      </div>
    </div>
  );
};
