import React from "react";
import { Sequence } from "remotion";
import { Scene3a_Decomposition } from "./Scene3a_Decomposition";
import { Scene3b_Assignment } from "./Scene3b_Assignment";
import { Scene3c_Verification } from "./Scene3c_Verification";
import { Scene3d_TrustSafety } from "./Scene3d_TrustSafety";
import { Scene3e_ParallelDAG } from "./Scene3e_ParallelDAG";

export const Scene3_Protocol: React.FC = () => {
  return (
    <div style={{ width: "100%", height: "100%", position: "relative" }}>
      <Sequence from={0} durationInFrames={90}>
        <Scene3a_Decomposition />
      </Sequence>
      <Sequence from={90} durationInFrames={90}>
        <Scene3b_Assignment />
      </Sequence>
      <Sequence from={180} durationInFrames={90}>
        <Scene3c_Verification />
      </Sequence>
      <Sequence from={270} durationInFrames={90}>
        <Scene3d_TrustSafety />
      </Sequence>
      <Sequence from={360} durationInFrames={90}>
        <Scene3e_ParallelDAG />
      </Sequence>
    </div>
  );
};
