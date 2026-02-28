import React from "react";
import { useCurrentFrame, interpolate, spring, useVideoConfig } from "remotion";
import { colors, fonts } from "../styles/theme";
import { TerminalLineType } from "../data/demoOutput";

interface Props {
  text: string;
  type: TerminalLineType;
  startFrame?: number;
}

const typeColors: Record<TerminalLineType, string> = {
  header: colors.grayDark,
  decompose: colors.primary,
  assign: colors.orange,
  execute: colors.yellow,
  complete: colors.green,
  verify_pass: colors.green,
  verify_fail: colors.red,
  trust: colors.secondary,
  reassign: colors.orange,
  result: colors.primary,
  blank: colors.white,
};

export const TerminalLine: React.FC<Props> = ({ text, type, startFrame = 0 }) => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  const opacity = interpolate(frame, [startFrame, startFrame + 4], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });

  const slideIn = spring({
    frame: frame - startFrame,
    fps,
    config: { damping: 20, mass: 0.5, stiffness: 200 },
  });

  const color = typeColors[type];

  return (
    <div
      style={{
        opacity,
        transform: `translateX(${(1 - slideIn) * 20}px)`,
        fontFamily: fonts.mono,
        fontSize: 18,
        color,
        lineHeight: 1.8,
        whiteSpace: "pre",
      }}
    >
      {type === "result" ? (
        <span style={{ fontWeight: 700 }}>{text}</span>
      ) : (
        text
      )}
    </div>
  );
};
