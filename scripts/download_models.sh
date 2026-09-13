#!/usr/bin/env bash
# Downloads the MediaPipe pose landmarker model used by src/pose_features.py.
# YOLOv8 weights (yolov8n.pt) are fetched automatically by ultralytics on
# first run, so they don't need a manual step here.
set -euo pipefail
mkdir -p "$(dirname "$0")/../models"
curl -sS -L -o "$(dirname "$0")/../models/pose_landmarker_lite.task" \
  "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
echo "Saved models/pose_landmarker_lite.task"
