import React from "react";
import { useCurrentFrame } from "remotion";
import { random } from "remotion";
import { colors } from "../styles/theme";

interface Props {
  count?: number;
  opacity?: number;
}

export const ParticleField: React.FC<Props> = ({ count = 80, opacity = 0.3 }) => {
  const frame = useCurrentFrame();

  const particles = Array.from({ length: count }, (_, i) => {
    const x = random(`px-${i}`) * 1920;
    const y = random(`py-${i}`) * 1080;
    const size = 1 + random(`ps-${i}`) * 3;
    const speed = 0.2 + random(`psp-${i}`) * 0.8;
    const phase = random(`pp-${i}`) * Math.PI * 2;
    const color = random(`pc-${i}`) > 0.7 ? colors.secondary : colors.primary;

    const animX = x + Math.sin(frame * 0.01 * speed + phase) * 30;
    const animY = y + Math.cos(frame * 0.008 * speed + phase) * 20;
    const animOpacity = opacity * (0.3 + 0.7 * Math.sin(frame * 0.02 * speed + phase) ** 2);

    return { x: animX, y: animY, size, color, opacity: animOpacity, key: i };
  });

  return (
    <div style={{ position: "absolute", inset: 0, overflow: "hidden" }}>
      {particles.map((p) => (
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
            boxShadow: `0 0 ${p.size * 3}px ${p.color}`,
          }}
        />
      ))}
    </div>
  );
};
