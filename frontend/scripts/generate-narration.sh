#!/bin/bash
# Generate narration audio for Delegato demo video using Microsoft Edge TTS
# Usage: bash frontend/scripts/generate-narration.sh

set -e

VOICE="en-US-AriaNeural"
RATE="+50%"
OUTPUT_DIR="$(dirname "$0")/../public/audio/narration"

mkdir -p "$OUTPUT_DIR"

echo "Generating narration with voice: $VOICE (rate: $RATE)"

# Scene 1: Logo Reveal (max 4.2s window)
echo "  [1/12] scene1.mp3"
edge-tts --voice "$VOICE" --rate "$RATE" \
  --text "Delegato! Smart AI agent delegation. Pure Python." \
  --write-media "$OUTPUT_DIR/scene1.mp3"

# Scene 2: The Problem (max 5.2s window)
echo "  [2/12] scene2.mp3"
edge-tts --voice "$VOICE" --rate "$RATE" \
  --text "The problem? Multiple agents, zero coordination. Just chaos." \
  --write-media "$OUTPUT_DIR/scene2.mp3"

# Scene 3a: DAG Decomposition (max 2.8s window)
echo "  [3/12] scene3a.mp3"
edge-tts --voice "$VOICE" --rate "$RATE" \
  --text "Step one -- decompose your goal into subtasks." \
  --write-media "$OUTPUT_DIR/scene3a.mp3"

# Scene 3b: Agent Scoring (max 3.2s window)
echo "  [4/12] scene3b.mp3"
edge-tts --voice "$VOICE" --rate "$RATE" \
  --text "Score each agent. Assign the best match." \
  --write-media "$OUTPUT_DIR/scene3b.mp3"

# Scene 3c: Verification (max 3.0s window)
echo "  [5/12] scene3c.mp3"
edge-tts --voice "$VOICE" --rate "$RATE" \
  --text "Multi-judge consensus verifies results." \
  --write-media "$OUTPUT_DIR/scene3c.mp3"

# Scene 3d: Trust & Circuit Breaker (max 3.0s window)
echo "  [6/12] scene3d.mp3"
edge-tts --voice "$VOICE" --rate "$RATE" \
  --text "Trust adapts. Too low? Circuit breaker." \
  --write-media "$OUTPUT_DIR/scene3d.mp3"

# Scene 3e: Parallel Execution (max 3.0s window)
echo "  [7/12] scene3e.mp3"
edge-tts --voice "$VOICE" --rate "$RATE" \
  --text "Parallel execution. Failures auto-retry." \
  --write-media "$OUTPUT_DIR/scene3e.mp3"

# Scene 4: Live Demo (max 10.0s window)
echo "  [8/12] scene4.mp3"
edge-tts --voice "$VOICE" --rate "$RATE" \
  --text "Watch a live pipeline! Decompose, assign, execute, verify. See that failure? Delegato retries and succeeds. Lightning fast, pennies to run." \
  --write-media "$OUTPUT_DIR/scene4.mp3"

# Scene 5: Architecture (max 7.0s window)
echo "  [9/12] scene5.mp3"
edge-tts --voice "$VOICE" --rate "$RATE" \
  --text "Six clean components. Tasks decompose, get scored, route to agents, and verified. Trust updates flow back." \
  --write-media "$OUTPUT_DIR/scene5.mp3"

# Scene 6: Benchmarks (max 8.0s window)
echo "  [10/12] scene6.mp3"
edge-tts --voice "$VOICE" --rate "$RATE" \
  --text "Two forty benchmarks, forty tasks. Delegato excels at medium difficulty -- where coordination matters. Keeps improving across trials." \
  --write-media "$OUTPUT_DIR/scene6.mp3"

# Scene 7: Code Showcase (max 5.3s window)
echo "  [11/12] scene7.mp3"
edge-tts --voice "$VOICE" --rate "$RATE" \
  --text "Getting started is easy. Define an agent, create a delegator, run. All built in." \
  --write-media "$OUTPUT_DIR/scene7.mp3"

# Scene 8: Closing (max 4.3s window)
echo "  [12/12] scene8.mp3"
edge-tts --voice "$VOICE" --rate "$RATE" \
  --text "Pip install delegato. Go build something awesome." \
  --write-media "$OUTPUT_DIR/scene8.mp3"

echo ""
echo "All narration files generated in: $OUTPUT_DIR"
