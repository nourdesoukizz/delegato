import React from "react";
import { useCurrentFrame, spring, useVideoConfig, interpolate } from "remotion";
import { colors, fonts } from "../styles/theme";
import { AnimatedBar } from "../components/AnimatedBar";
import { Counter } from "../components/Counter";
import { TrustGraph } from "../components/TrustGraph";
import { byDifficulty, trialProgression, overallStats } from "../data/benchmarkData";

export const Scene6_Benchmarks: React.FC = () => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  const titleOpacity = interpolate(frame, [0, 15], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });

  return (
    <div style={{ width: "100%", height: "100%", position: "relative" }}>
      {/* Title */}
      <div
        style={{
          position: "absolute",
          top: 30,
          width: "100%",
          textAlign: "center",
          opacity: titleOpacity,
        }}
      >
        <span style={{ fontSize: 36, fontWeight: 700, color: colors.white, fontFamily: fonts.sans }}>
          Benchmark Results
        </span>
        <div style={{ fontSize: 16, color: colors.gray, fontFamily: fonts.mono, marginTop: 8 }}>
          40 tasks × 3 trials = 240 runs
        </div>
      </div>

      {/* Bar chart by difficulty */}
      <div style={{ position: "absolute", top: 130, left: 100, width: 800 }}>
        <div style={{ fontSize: 18, fontWeight: 600, color: colors.gray, fontFamily: fonts.sans, marginBottom: 20 }}>
          Success Rate by Difficulty
        </div>
        {byDifficulty.map((d, i) => (
          <div key={i} style={{ marginBottom: 24 }}>
            <div
              style={{
                fontSize: 14,
                color: colors.gray,
                fontFamily: fonts.mono,
                marginBottom: 6,
                opacity: interpolate(frame, [15 + i * 12, 25 + i * 12], [0, 1], {
                  extrapolateLeft: "clamp",
                  extrapolateRight: "clamp",
                }),
              }}
            >
              {d.label}
            </div>
            <div style={{ display: "flex", gap: 12, alignItems: "center" }}>
              <div style={{ width: 60, fontSize: 12, color: colors.gray, fontFamily: fonts.mono, textAlign: "right" }}>
                Naive
              </div>
              <AnimatedBar
                label=""
                value={d.naive}
                color={colors.grayDark}
                startFrame={20 + i * 12}
                width={300}
                height={20}
                showValue={true}
              />
            </div>
            <div style={{ display: "flex", gap: 12, alignItems: "center", marginTop: 4 }}>
              <div style={{ width: 60, fontSize: 12, color: colors.primary, fontFamily: fonts.mono, textAlign: "right" }}>
                Delegato
              </div>
              <AnimatedBar
                label=""
                value={d.delegato}
                color={d.delegato > d.naive ? colors.green : d.delegato < d.naive ? colors.red : colors.primary}
                startFrame={24 + i * 12}
                width={300}
                height={20}
                showValue={true}
              />
            </div>
          </div>
        ))}

        {/* Medium task highlight */}
        <div
          style={{
            marginTop: 20,
            padding: "12px 20px",
            backgroundColor: `${colors.green}15`,
            border: `1px solid ${colors.green}44`,
            borderRadius: 8,
            display: "inline-block",
            opacity: interpolate(frame, [80, 95], [0, 1], {
              extrapolateLeft: "clamp",
              extrapolateRight: "clamp",
            }),
          }}
        >
          <span style={{ fontFamily: fonts.mono, fontSize: 18, color: colors.green, fontWeight: 700 }}>
            Delegato outperforms on medium tasks
          </span>
        </div>
      </div>

      {/* Right side: Stats + Trial progression */}
      <div style={{ position: "absolute", top: 130, right: 100, width: 700 }}>
        {/* Overall result */}
        <div
          style={{
            textAlign: "center",
            marginBottom: 40,
            padding: "16px 32px",
            backgroundColor: `${colors.green}12`,
            border: `1px solid ${colors.green}33`,
            borderRadius: 12,
            opacity: interpolate(frame, [10, 25], [0, 1], {
              extrapolateLeft: "clamp",
              extrapolateRight: "clamp",
            }),
          }}
        >
          <span style={{ fontFamily: fonts.sans, fontSize: 28, fontWeight: 700, color: colors.green }}>
            Delegato outperforms naive execution
          </span>
          <div style={{ fontSize: 14, color: colors.gray, fontFamily: fonts.mono, marginTop: 8 }}>
            240 benchmark runs across 40 tasks
          </div>
        </div>

        {/* Trial progression chart */}
        <div style={{ marginBottom: 16 }}>
          <span style={{ fontFamily: fonts.sans, fontSize: 16, color: colors.gray, fontWeight: 600 }}>
            Trial Progression
          </span>
        </div>
        <TrustGraph
          data={trialProgression.map((t) => ({ label: t.trial.replace("Trial ", "T"), value: t.delegato }))}
          startFrame={50}
          width={650}
          height={250}
          color={colors.primary}
        />

        {/* Cost info */}
        <div
          style={{
            marginTop: 20,
            fontFamily: fonts.mono,
            fontSize: 14,
            color: colors.grayDark,
            opacity: interpolate(frame, [120, 135], [0, 1], {
              extrapolateLeft: "clamp",
              extrapolateRight: "clamp",
            }),
          }}
        >
          Cost: {overallStats.costMultiplier}x | Latency: {overallStats.avgLatencyDelegato}s avg
        </div>
      </div>
    </div>
  );
};
