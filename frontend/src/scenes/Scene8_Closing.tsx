import React from "react";
import { useCurrentFrame, spring, useVideoConfig, interpolate } from "remotion";
import { colors, fonts } from "../styles/theme";
import { GlowBox } from "../components/GlowBox";

export const Scene8_Closing: React.FC = () => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  const logoScale = spring({
    frame: frame - 5,
    fps,
    config: { damping: 12, mass: 0.8, stiffness: 180 },
  });

  const logoOpacity = interpolate(frame, [5, 20], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });

  const fadeOut = interpolate(frame, [120, 150], [1, 0], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });

  // Particle explosion effect
  const explosionParticles = Array.from({ length: 40 }, (_, i) => {
    const angle = (i / 40) * Math.PI * 2;
    const speed = 2 + (i % 5) * 1.5;
    const explosionFrame = 80;
    const elapsed = Math.max(0, frame - explosionFrame);
    const distance = elapsed * speed;
    const opacity = Math.max(0, 1 - elapsed / 40);
    const x = 960 + Math.cos(angle) * distance;
    const y = 540 + Math.sin(angle) * distance;
    const size = 2 + (i % 3);
    const color = i % 3 === 0 ? colors.primary : i % 3 === 1 ? colors.secondary : colors.green;
    return { x, y, size, opacity, color, key: i };
  });

  return (
    <div
      style={{
        width: "100%",
        height: "100%",
        position: "relative",
        opacity: fadeOut,
      }}
    >
      {/* Explosion particles */}
      {frame >= 80 &&
        explosionParticles.map((p) => (
          <div
            key={p.key}
            style={{
              position: "absolute",
              left: p.x,
              top: p.y,
              width: p.size,
              height: p.size,
              borderRadius: "50%",
              backgroundColor: p.color,
              opacity: p.opacity,
              boxShadow: `0 0 ${p.size * 4}px ${p.color}`,
            }}
          />
        ))}

      {/* Logo */}
      <div
        style={{
          position: "absolute",
          top: 200,
          width: "100%",
          textAlign: "center",
          opacity: logoOpacity,
          transform: `scale(${logoScale})`,
        }}
      >
        <span
          style={{
            fontSize: 100,
            fontWeight: 700,
            fontFamily: fonts.sans,
            background: `linear-gradient(135deg, ${colors.primary}, ${colors.secondary})`,
            WebkitBackgroundClip: "text",
            WebkitTextFillColor: "transparent",
          }}
        >
          delegato
        </span>
      </div>

      {/* pip install box */}
      <div
        style={{
          position: "absolute",
          top: 380,
          left: "50%",
          transform: "translateX(-50%)",
          opacity: interpolate(frame, [25, 40], [0, 1], {
            extrapolateLeft: "clamp",
            extrapolateRight: "clamp",
          }),
        }}
      >
        <GlowBox color={colors.primary} padding="16px 48px" borderRadius={12}>
          <div style={{ fontFamily: fonts.mono, fontSize: 28, color: colors.primary, fontWeight: 600 }}>
            <span style={{ color: colors.gray }}>$ </span>pip install delegato
          </div>
        </GlowBox>
      </div>

      {/* GitHub URL */}
      <div
        style={{
          position: "absolute",
          top: 490,
          width: "100%",
          textAlign: "center",
          opacity: interpolate(frame, [40, 55], [0, 1], {
            extrapolateLeft: "clamp",
            extrapolateRight: "clamp",
          }),
        }}
      >
        <span style={{ fontFamily: fonts.mono, fontSize: 20, color: colors.gray }}>
          github.com/nourspace/delegato
        </span>
      </div>

      {/* DeepMind reference */}
      <div
        style={{
          position: "absolute",
          top: 560,
          width: "100%",
          textAlign: "center",
          opacity: interpolate(frame, [50, 65], [0, 1], {
            extrapolateLeft: "clamp",
            extrapolateRight: "clamp",
          }),
        }}
      >
        <span
          style={{
            fontFamily: fonts.sans,
            fontSize: 16,
            color: colors.grayDark,
            fontStyle: "italic",
          }}
        >
          Based on DeepMind's Delegation Protocol (arXiv:2602.11865)
        </span>
      </div>

      {/* Bottom tagline */}
      <div
        style={{
          position: "absolute",
          bottom: 100,
          width: "100%",
          textAlign: "center",
          opacity: interpolate(frame, [60, 75], [0, 1], {
            extrapolateLeft: "clamp",
            extrapolateRight: "clamp",
          }),
        }}
      >
        <span
          style={{
            fontFamily: fonts.sans,
            fontSize: 24,
            color: colors.white,
            fontWeight: 300,
          }}
        >
          Intelligent delegation for multi-agent systems
        </span>
      </div>
    </div>
  );
};
