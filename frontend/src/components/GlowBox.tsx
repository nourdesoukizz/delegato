import React from "react";
import { useCurrentFrame } from "remotion";
import { colors } from "../styles/theme";

interface Props {
  children: React.ReactNode;
  color?: string;
  borderRadius?: number;
  padding?: string;
  style?: React.CSSProperties;
}

export const GlowBox: React.FC<Props> = ({
  children,
  color = colors.primary,
  borderRadius = 16,
  padding = "32px",
  style,
}) => {
  const frame = useCurrentFrame();
  const glowIntensity = 0.3 + 0.2 * Math.sin(frame * 0.05);

  return (
    <div
      style={{
        backgroundColor: `${color}0D`,
        border: `1px solid ${color}44`,
        borderRadius,
        padding,
        boxShadow: `0 0 ${30 * glowIntensity}px ${color}33, inset 0 0 ${20 * glowIntensity}px ${color}11`,
        ...style,
      }}
    >
      {children}
    </div>
  );
};
