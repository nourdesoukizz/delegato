import React from "react";
import { useCurrentFrame, spring, useVideoConfig, interpolate } from "remotion";
import { colors, fonts } from "../styles/theme";
import { CodeBlock } from "../components/CodeBlock";

const quickStartCode = `from delegato import Agent, Delegator, Task
from delegato import VerificationSpec, VerificationMethod

async def my_handler(task):
    return TaskResult(
        task_id=task.id,
        agent_id="worker",
        output="Hello from delegato!",
        success=True,
    )

agent = Agent(
    id="worker",
    capabilities=["general"],
    handler=my_handler,
)

delegator = Delegator(
    agents=[agent],
    llm_call=mock_llm,
)

result = await delegator.run(task)`;

const features = [
  { text: "DAG-based task decomposition", delay: 10 },
  { text: "Multi-objective agent scoring", delay: 20 },
  { text: "5 verification methods", delay: 30 },
  { text: "Multi-judge consensus", delay: 40 },
  { text: "Trust tracking + circuit breaker", delay: 50 },
  { text: "Automatic retry & fallback", delay: 60 },
  { text: "Full audit logging", delay: 70 },
  { text: "310 tests, 94% coverage", delay: 80 },
];

export const Scene7_CodeShowcase: React.FC = () => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  const titleOpacity = interpolate(frame, [0, 12], [0, 1], {
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
          Quick Start
        </span>
      </div>

      {/* Left: Code */}
      <div style={{ position: "absolute", top: 90, left: 60, width: 920 }}>
        <CodeBlock code={quickStartCode} startFrame={5} lineDelay={2} fontSize={17} />
      </div>

      {/* Right: Features checklist */}
      <div style={{ position: "absolute", top: 90, right: 60, width: 800 }}>
        <div
          style={{
            backgroundColor: `${colors.primary}08`,
            border: `1px solid ${colors.grayDarker}`,
            borderRadius: 12,
            padding: "28px 32px",
          }}
        >
          <div
            style={{
              fontSize: 22,
              fontWeight: 700,
              color: colors.primary,
              fontFamily: fonts.sans,
              marginBottom: 20,
              opacity: interpolate(frame, [5, 15], [0, 1], {
                extrapolateLeft: "clamp",
                extrapolateRight: "clamp",
              }),
            }}
          >
            Features
          </div>
          {features.map((f, i) => {
            const checkSpring = spring({
              frame: frame - f.delay,
              fps,
              config: { damping: 15, mass: 0.5, stiffness: 200 },
            });
            const featureOpacity = interpolate(
              frame,
              [f.delay, f.delay + 8],
              [0, 1],
              { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
            );

            return (
              <div
                key={i}
                style={{
                  display: "flex",
                  alignItems: "center",
                  gap: 14,
                  marginBottom: 14,
                  opacity: featureOpacity,
                  transform: `translateX(${(1 - checkSpring) * 20}px)`,
                }}
              >
                <div
                  style={{
                    width: 24,
                    height: 24,
                    borderRadius: 6,
                    backgroundColor: `${colors.green}22`,
                    border: `1px solid ${colors.green}66`,
                    display: "flex",
                    alignItems: "center",
                    justifyContent: "center",
                    transform: `scale(${checkSpring})`,
                  }}
                >
                  <span style={{ color: colors.green, fontSize: 14, fontWeight: 700 }}>✓</span>
                </div>
                <span style={{ fontFamily: fonts.sans, fontSize: 18, color: colors.white }}>
                  {f.text}
                </span>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
};
