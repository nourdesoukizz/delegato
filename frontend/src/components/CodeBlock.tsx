import React from "react";
import { useCurrentFrame, interpolate, spring, useVideoConfig } from "remotion";
import { colors, fonts } from "../styles/theme";

interface Props {
  code: string;
  startFrame?: number;
  lineDelay?: number;
  fontSize?: number;
  style?: React.CSSProperties;
}

const KEYWORDS = [
  "import", "from", "async", "await", "def", "return", "if", "else",
  "class", "True", "False", "None", "print", "global",
];
const DECORATORS = ["@"];
const STRINGS = /(["'])(?:(?!\1|\\).|\\.)*\1/g;
const COMMENTS = /#.*/g;

const highlightLine = (line: string): React.ReactNode[] => {
  const parts: React.ReactNode[] = [];
  let remaining = line;
  let key = 0;

  // Simple syntax highlighting
  const tokens = remaining.split(/(\s+|[()[\]{},=.:@]|"[^"]*"|'[^']*'|#.*)/);
  for (const token of tokens) {
    if (!token) continue;
    let color: string = colors.white;
    if (KEYWORDS.includes(token)) color = colors.secondary;
    else if (token.startsWith("#")) color = colors.grayDark;
    else if (token.startsWith('"') || token.startsWith("'")) color = colors.green;
    else if (token.startsWith("@")) color = colors.orange;
    else if (/^\d+(\.\d+)?$/.test(token)) color = colors.orange;
    else if (/^[()[\]{},=.:]$/.test(token)) color = colors.gray;
    parts.push(
      <span key={key++} style={{ color }}>
        {token}
      </span>
    );
  }
  return parts;
};

export const CodeBlock: React.FC<Props> = ({
  code,
  startFrame = 0,
  lineDelay = 3,
  fontSize = 20,
  style,
}) => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();
  const lines = code.split("\n");

  return (
    <div
      style={{
        backgroundColor: "rgba(10, 14, 26, 0.9)",
        borderRadius: 12,
        padding: "24px 28px",
        border: `1px solid ${colors.grayDarker}`,
        fontFamily: fonts.mono,
        fontSize,
        lineHeight: 1.6,
        overflow: "hidden",
        ...style,
      }}
    >
      {lines.map((line, i) => {
        const lineStart = startFrame + i * lineDelay;
        const slideIn = spring({
          frame: frame - lineStart,
          fps,
          config: { damping: 20, mass: 0.5, stiffness: 200 },
        });
        const opacity = interpolate(frame, [lineStart, lineStart + 5], [0, 1], {
          extrapolateLeft: "clamp",
          extrapolateRight: "clamp",
        });

        return (
          <div
            key={i}
            style={{
              opacity,
              transform: `translateX(${(1 - slideIn) * 30}px)`,
              whiteSpace: "pre",
            }}
          >
            <span style={{ color: colors.grayDark, marginRight: 16, userSelect: "none" }}>
              {String(i + 1).padStart(2, " ")}
            </span>
            {highlightLine(line)}
          </div>
        );
      })}
    </div>
  );
};
