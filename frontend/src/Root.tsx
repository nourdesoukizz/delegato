import React from "react";
import { Composition } from "remotion";
import { DelegatoDemo } from "./video/DelegatoDemo";

export const Root: React.FC = () => {
  return (
    <Composition
      id="DelegatoDemo"
      component={DelegatoDemo}
      durationInFrames={1800}
      fps={30}
      width={1920}
      height={1080}
    />
  );
};
