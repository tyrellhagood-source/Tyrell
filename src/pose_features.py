"""Per-player pose extraction with MediaPipe (Tasks API).

For every frame a player was tracked in (see track.py), this crops their
bounding box (with a little padding) and runs MediaPipe's PoseLandmarker on
just that crop. Landmarks are mapped back into full-frame pixel coordinates
and attached to the matching TrackPoint.

This is intentionally best-effort: on a small, blurry, or heavily-occluded
crop MediaPipe may fail to find a confident pose, in which case that frame
just has keypoints=None and the trajectory-based metrics (which don't need
pose at all) still work.
"""
import os

import cv2
import mediapipe as mp
import numpy as np

from schema import Track, TrackPoint

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "pose_landmarker_lite.task")

BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# A subset of the 33 BlazePose landmark indices we actually use.
LM = {
    "left_shoulder": 11, "right_shoulder": 12,
    "left_hip": 23, "right_hip": 24,
    "left_knee": 25, "right_knee": 26,
    "left_ankle": 27, "right_ankle": 28,
}

PAD_FRAC = 0.15  # pad the crop so limbs near the box edge aren't cut off


def _make_landmarker(model_path: str = MODEL_PATH):
    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=VisionRunningMode.IMAGE,
        num_poses=1,
        min_pose_detection_confidence=0.4,
        min_pose_presence_confidence=0.4,
    )
    return PoseLandmarker.create_from_options(options)


def add_pose_to_track(video_path: str, track: Track, model_path: str = MODEL_PATH) -> Track:
    """Mutates and returns `track`, filling in `.keypoints` on each TrackPoint."""
    landmarker = _make_landmarker(model_path)
    cap = cv2.VideoCapture(video_path)
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    for point in track.points:
        cap.set(cv2.CAP_PROP_POS_FRAMES, point.frame)
        ok, frame = cap.read()
        if not ok:
            continue

        x1, y1, x2, y2 = point.bbox
        w, h = x2 - x1, y2 - y1
        x1 = max(0, int(x1 - w * PAD_FRAC))
        y1 = max(0, int(y1 - h * PAD_FRAC))
        x2 = min(frame_w, int(x2 + w * PAD_FRAC))
        y2 = min(frame_h, int(y2 + h * PAD_FRAC))
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            continue

        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
        try:
            result = landmarker.detect(mp_image)
        except Exception:
            continue

        if not result.pose_landmarks:
            continue

        landmarks = result.pose_landmarks[0]
        crop_h, crop_w = crop.shape[:2]
        keypoints = {}
        for name, idx in LM.items():
            lm = landmarks[idx]
            keypoints[name] = (x1 + lm.x * crop_w, y1 + lm.y * crop_h, lm.visibility)
        point.keypoints = keypoints

    cap.release()
    landmarker.close()
    return track


def torso_lean_angle(point: TrackPoint):
    """Angle (degrees) of the shoulder->hip vector vs vertical. ~0 = upright,
    larger = leaning forward/back (rough posture signal, e.g. sprint lean vs
    backpedal). Returns None if keypoints weren't found for this frame."""
    if not point.keypoints:
        return None
    kp = point.keypoints
    try:
        sx = (kp["left_shoulder"][0] + kp["right_shoulder"][0]) / 2
        sy = (kp["left_shoulder"][1] + kp["right_shoulder"][1]) / 2
        hx = (kp["left_hip"][0] + kp["right_hip"][0]) / 2
        hy = (kp["left_hip"][1] + kp["right_hip"][1]) / 2
    except KeyError:
        return None
    dx, dy = sx - hx, sy - hy
    if dx == 0 and dy == 0:
        return None
    angle = np.degrees(np.arctan2(abs(dx), abs(dy)))
    return float(angle)
