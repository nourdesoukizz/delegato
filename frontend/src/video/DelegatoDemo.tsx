import React from "react";
import { Audio, Sequence, staticFile, useCurrentFrame, interpolate } from "remotion";
import { ParticleField } from "../components/ParticleField";
import { Scene1_LogoReveal } from "../scenes/Scene1_LogoReveal";
import { Scene2_TheProblem } from "../scenes/Scene2_TheProblem";
import { Scene3_Protocol } from "../scenes/Scene3_Protocol";
import { Scene4_LiveDemo } from "../scenes/Scene4_LiveDemo";
import { Scene5_Architecture } from "../scenes/Scene5_Architecture";
import { Scene6_Benchmarks } from "../scenes/Scene6_Benchmarks";
import { Scene7_CodeShowcase } from "../scenes/Scene7_CodeShowcase";
import { Scene8_Closing } from "../scenes/Scene8_Closing";
import { colors } from "../styles/theme";

export const DelegatoDemo: React.FC = () => {
  const frame = useCurrentFrame();

  return (
    <div
      style={{
        width: 1920,
        height: 1080,
        backgroundColor: colors.bg,
        position: "relative",
        overflow: "hidden",
      }}
    >
      {/* Background music — lower volume to sit behind narration */}
      <Audio
        src={staticFile("audio/background-music.mp3")}
        volume={(f) =>
          interpolate(f, [0, 90, 1650, 1800], [0, 0.4, 0.4, 0], {
            extrapolateLeft: "clamp",
            extrapolateRight: "clamp",
          })
        }
      />

      {/* Narration audio — one clip per scene */}
      <Sequence from={30} durationInFrames={120}>
        <Audio src={staticFile("audio/narration/scene1.mp3")} volume={1} />
      </Sequence>
      <Sequence from={155} durationInFrames={145}>
        <Audio src={staticFile("audio/narration/scene2.mp3")} volume={1} />
      </Sequence>
      <Sequence from={310} durationInFrames={80}>
        <Audio src={staticFile("audio/narration/scene3a.mp3")} volume={1} />
      </Sequence>
      <Sequence from={395} durationInFrames={100}>
        <Audio src={staticFile("audio/narration/scene3b.mp3")} volume={1} />
      </Sequence>
      <Sequence from={490} durationInFrames={90}>
        <Audio src={staticFile("audio/narration/scene3c.mp3")} volume={1} />
      </Sequence>
      <Sequence from={580} durationInFrames={90}>
        <Audio src={staticFile("audio/narration/scene3d.mp3")} volume={1} />
      </Sequence>
      <Sequence from={670} durationInFrames={80}>
        <Audio src={staticFile("audio/narration/scene3e.mp3")} volume={1} />
      </Sequence>
      <Sequence from={760} durationInFrames={295}>
        <Audio src={staticFile("audio/narration/scene4.mp3")} volume={1} />
      </Sequence>
      <Sequence from={1060} durationInFrames={200}>
        <Audio src={staticFile("audio/narration/scene5.mp3")} volume={1} />
      </Sequence>
      <Sequence from={1270} durationInFrames={220}>
        <Audio src={staticFile("audio/narration/scene6.mp3")} volume={1} />
      </Sequence>
      <Sequence from={1510} durationInFrames={130}>
        <Audio src={staticFile("audio/narration/scene7.mp3")} volume={1} />
      </Sequence>
      <Sequence from={1670} durationInFrames={80}>
        <Audio src={staticFile("audio/narration/scene8.mp3")} volume={1} />
      </Sequence>

      {/* Persistent particle background */}
      <ParticleField count={80} opacity={0.15} />

      {/* Scene 1: Logo Reveal — 0-5s (frames 0-149) */}
      <Sequence from={0} durationInFrames={150}>
        <Scene1_LogoReveal />
      </Sequence>

      {/* Scene 2: The Problem — 5-10s (frames 150-299) */}
      <Sequence from={150} durationInFrames={150}>
        <Scene2_TheProblem />
      </Sequence>

      {/* Scene 3: Protocol — 10-25s (frames 300-749) */}
      <Sequence from={300} durationInFrames={450}>
        <Scene3_Protocol />
      </Sequence>

      {/* Scene 4: Live Demo — 25-35s (frames 750-1049) */}
      <Sequence from={750} durationInFrames={300}>
        <Scene4_LiveDemo />
      </Sequence>

      {/* Scene 5: Architecture — 35-42s (frames 1050-1259) */}
      <Sequence from={1050} durationInFrames={210}>
        <Scene5_Architecture />
      </Sequence>

      {/* Scene 6: Benchmarks — 42-50s (frames 1260-1499) */}
      <Sequence from={1260} durationInFrames={240}>
        <Scene6_Benchmarks />
      </Sequence>

      {/* Scene 7: Code Showcase — 50-55s (frames 1500-1649) */}
      <Sequence from={1500} durationInFrames={150}>
        <Scene7_CodeShowcase />
      </Sequence>

      {/* Scene 8: Closing — 55-60s (frames 1650-1799) */}
      <Sequence from={1650} durationInFrames={150}>
        <Scene8_Closing />
      </Sequence>

      {/* Scene transition overlays — subtle fade between scenes */}
      {[150, 300, 750, 1050, 1260, 1500, 1650].map((transitionFrame) => {
        const transOpacity = interpolate(
          frame,
          [transitionFrame - 5, transitionFrame, transitionFrame + 5],
          [0, 0.3, 0],
          { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
        );
        return (
          <div
            key={transitionFrame}
            style={{
              position: "absolute",
              inset: 0,
              backgroundColor: colors.bg,
              opacity: transOpacity,
              pointerEvents: "none",
            }}
          />
        );
      })}
    </div>
  );
};
