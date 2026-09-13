"""Assisted player identification.

Full automatic "who is this athlete" recognition (jersey OCR / face ID) is not
reliable on typical drill/broadcast footage without a trained, roster-specific
model, so this MVP uses a fast assisted flow instead:

1. track.py detects + tracks every person in the clip and gives each a raw
   track ID (TID).
2. save_track_thumbnails() grabs one crop per track so a human can see, at a
   glance, which TID is the WR and which is the CB (and ignore any bystanders
   the detector picked up).
3. The human supplies a small PID map (track_id -> role/name/number) below.
4. That PID then persists in the report even if the raw tracker re-assigns a
   new TID mid-clip (e.g. after a brief occlusion) -- the human just re-points
   the new TID at the same PID.
"""
import os

import cv2

from schema import PlayerIdentity


def save_track_thumbnails(video_path: str, tracks: dict, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)

    saved = {}
    for tid, track in tracks.items():
        if not track.points:
            continue
        # pick a mid-clip frame so the crop isn't a partially-entered player
        mid = track.points[len(track.points) // 2]
        cap.set(cv2.CAP_PROP_POS_FRAMES, mid.frame)
        ok, frame = cap.read()
        if not ok:
            continue
        x1, y1, x2, y2 = [int(v) for v in mid.bbox]
        x1, y1 = max(x1, 0), max(y1, 0)
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            continue
        path = os.path.join(out_dir, f"track_{tid}.jpg")
        cv2.imwrite(path, crop)
        saved[tid] = path

    cap.release()
    return saved


def build_identity_map(assignments: list) -> dict:
    """assignments: list of dicts like
       {"track_id": 3, "role": "WR", "name": "J. Smith", "number": "11"}
    Returns dict[track_id] -> PlayerIdentity
    """
    return {
        a["track_id"]: PlayerIdentity(
            track_id=a["track_id"],
            role=a["role"],
            name=a.get("name", a["role"]),
            number=a.get("number"),
            team=a.get("team"),
        )
        for a in assignments
    }
