import React from "react";
import { useCurrentFrame, interpolate } from "remotion";
import { fonts } from "../styles/theme";

interface Props {
  text: string;
  startFrame?: number;
  charsPerFrame?: number;
  fontSize?: number;
  color?: string;
  fontFamily?: string;
  showCursor?: boolean;
  style?: React.CSSProperties;
}

export const TypewriterText: React.FC<Props> = ({
  text,
  startFrame = 0,
  charsPerFrame = 0.8,
  fontSize = 32,
  color = "#F8FAFC",
  fontFamily = fonts.mono,
  showCursor = true,
  style,
}) => {
  const frame = useCurrentFrame();
  const elapsed = Math.max(0, frame - startFrame);
  const charsVisible = Math.min(Math.floor(elapsed * charsPerFrame), text.length);
  const displayText = text.slice(0, charsVisible);
  const cursorOpacity = showCursor && charsVisible < text.length
    ? Math.round(frame * 0.06) % 2 === 0 ? 1 : 0
    : 0;

  return (
    <span
      style={{
        fontSize,
        color,
        fontFamily,
        whiteSpace: "pre",
        ...style,
      }}
    >
      {displayText}
      {showCursor && (
        <span style={{ opacity: cursorOpacity, color }}>▌</span>
      )}
    </span>
  );
};
