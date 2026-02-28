import React from "react";
import { useCurrentFrame, interpolate } from "remotion";
import { colors, fonts } from "../styles/theme";
import { TerminalLine } from "../components/TerminalLine";
import { demoLines } from "../data/demoOutput";

export const Scene4_LiveDemo: React.FC = () => {
  const frame = useCurrentFrame();
  const LINE_DELAY = 8; // frames between each line

  const titleOpacity = interpolate(frame, [0, 15], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });

  const visibleLines = Math.min(
    Math.floor(frame / LINE_DELAY),
    demoLines.length
  );

  // Auto-scroll: offset to show recent lines
  const maxVisibleLines = 16;
  const scrollOffset = Math.max(0, visibleLines - maxVisibleLines);

  return (
    <div style={{ width: "100%", height: "100%", position: "relative" }}>
      {/* Title */}
      <div
        style={{
          position: "absolute",
          top: 24,
          left: 60,
          opacity: titleOpacity,
          display: "flex",
          alignItems: "center",
          gap: 12,
        }}
      >
        <span style={{ fontSize: 28, fontWeight: 600, color: colors.white, fontFamily: fonts.sans }}>
          Live Demo
        </span>
        <span style={{ fontSize: 18, color: colors.gray, fontFamily: fonts.mono }}>
          research_pipeline.py
        </span>
      </div>

      {/* Terminal window */}
      <div
        style={{
          position: "absolute",
          top: 80,
          left: 80,
          right: 80,
          bottom: 40,
          backgroundColor: "#0D1117",
          borderRadius: 12,
          border: `1px solid ${colors.grayDarker}`,
          overflow: "hidden",
        }}
      >
        {/* Terminal title bar */}
        <div
          style={{
            height: 36,
            backgroundColor: "#161B22",
            display: "flex",
            alignItems: "center",
            padding: "0 16px",
            gap: 8,
            borderBottom: `1px solid ${colors.grayDarker}`,
          }}
        >
          <div style={{ width: 12, height: 12, borderRadius: "50%", backgroundColor: "#FF5F57" }} />
          <div style={{ width: 12, height: 12, borderRadius: "50%", backgroundColor: "#FEBC2E" }} />
          <div style={{ width: 12, height: 12, borderRadius: "50%", backgroundColor: "#28C840" }} />
          <span
            style={{
              marginLeft: 12,
              fontFamily: fonts.mono,
              fontSize: 12,
              color: colors.grayDark,
            }}
          >
            python examples/research_pipeline.py
          </span>
        </div>

        {/* Terminal content */}
        <div style={{ padding: "16px 24px", overflow: "hidden" }}>
          {demoLines.slice(scrollOffset, visibleLines).map((line, i) => {
            const actualIndex = scrollOffset + i;
            return (
              <TerminalLine
                key={actualIndex}
                text={line.text}
                type={line.type}
                startFrame={actualIndex * LINE_DELAY}
              />
            );
          })}

          {/* Blinking cursor */}
          <div
            style={{
              fontFamily: fonts.mono,
              fontSize: 18,
              color: colors.green,
              opacity: Math.round(frame * 0.04) % 2 === 0 ? 1 : 0,
              marginTop: 4,
            }}
          >
            ▌
          </div>
        </div>
      </div>
    </div>
  );
};
