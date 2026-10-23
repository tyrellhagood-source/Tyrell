"""Player detection + multi-object tracking.

Uses a pretrained YOLOv8 person detector combined with a multi-object
tracker (both bundled in ultralytics) to assign each person in the clip a
persistent track ID (TID) across frames -- this is the piece that answers
"can we at least track the two athletes in a 1-on-1 rep".

Default tracker is BoT-SORT, not ByteTrack: it folds in a camera-motion
compensation step on top of ByteTrack's motion matching, which holds IDs
together noticeably better through the kind of contact/occlusion that's
constant in football and basketball (two players colliding at a break, a
screen, a pile). ByteTrack is still available (faster, slightly more ID
switches) via `tracker="bytetrack.yaml"` for anyone who wants to compare
the two on their own footage.

For genuinely crowded scenes -- a huddle, a sideline cluster, a pile where
players fully occlude each other for more than a frame or two -- plain
motion/IoU matching (what both of the above do by default) can still lose
or swap IDs, because there's no visual signal distinguishing who's who
once two boxes merge. config/botsort_crowd.yaml turns on BoT-SORT's
appearance (ReID) matching and a longer lost-track buffer for exactly that
case; pass tracker=resolve_tracker("botsort-crowd") to use it. Slower
(an extra model pass per box) and not independently verified against real
huddle footage -- see that file's own comments.

No training required: YOLOv8's COCO-pretrained weights already detect the
"person" class well enough for this use case. Jersey-number reading / true
player identification is NOT attempted here (that needs OCR on a much
higher-res crop and is unreliable on broadcast/phone footage) -- instead we
expose the raw tracks so a human assigns a PID (WR/CB + name) to each TID
once via identity.py. That PID then persists even if the underlying TID
resets (e.g. a player is briefly occluded and the tracker gives them a new
ID) as long as the human re-links it in the identity map.
"""
import os
from collections import defaultdict

from ultralytics import YOLO

from schema import Track, TrackPoint

PERSON_CLASS_ID = 0
CONFIG_DIR = os.path.join(os.path.dirname(__file__), "..", "config")

TRACKER_PRESETS = {
    "botsort": "botsort.yaml",       # ultralytics built-in, GMC only
    "bytetrack": "bytetrack.yaml",   # ultralytics built-in, fastest
    "botsort-crowd": os.path.join(CONFIG_DIR, "botsort_crowd.yaml"),  # + ReID, for huddles/piles
}


def resolve_tracker(name: str) -> str:
    """Maps a short preset name to what ultralytics' model.track(tracker=...)
    expects: either one of its own built-in config names, or a full path to
    one of ours (config/botsort_crowd.yaml)."""
    if name in TRACKER_PRESETS:
        return TRACKER_PRESETS[name]
    return name  # already a path/ultralytics name -- pass through


def run_tracking(video_path: str, model_path: str = "yolov8n.pt",
                  conf: float = 0.35, tracker: str = "botsort.yaml"):
    """Runs YOLO+BoT-SORT (or ByteTrack, if requested) over the whole video.

    Returns:
        tracks: dict[int, Track] keyed by track_id
        fps: float
        frame_size: (width, height)
    """
    model = YOLO(model_path)
    tracks_raw = defaultdict(list)

    results_gen = model.track(
        source=video_path,
        classes=[PERSON_CLASS_ID],
        conf=conf,
        tracker=tracker,
        persist=True,
        stream=True,
        verbose=False,
    )

    frame_size = None
    frame_idx = 0
    for result in results_gen:
        h, w = result.orig_shape
        frame_size = (w, h)

        boxes = result.boxes
        if boxes is None or boxes.id is None:
            frame_idx += 1
            continue

        ids = boxes.id.int().tolist()
        xyxy = boxes.xyxy.tolist()
        for tid, box in zip(ids, xyxy):
            x1, y1, x2, y2 = box
            cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
            tracks_raw[tid].append((frame_idx, (x1, y1, x2, y2), (cx, cy)))
        frame_idx += 1

    # Recover true fps from the source video (ultralytics doesn't expose it directly per-frame).
    import cv2
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.release()

    tracks = {}
    for tid, pts in tracks_raw.items():
        track_points = [
            TrackPoint(frame=f, t=f / fps, bbox=b, centroid=c)
            for f, b, c in sorted(pts, key=lambda p: p[0])
        ]
        tracks[tid] = Track(track_id=tid, points=track_points)

    return tracks, fps, frame_size


def tracks_to_json(tracks: dict, fps: float, frame_size) -> dict:
    return {
        "fps": fps,
        "frame_size": frame_size,
        "tracks": {
            str(tid): [
                {"frame": p.frame, "t": p.t, "bbox": list(p.bbox), "centroid": list(p.centroid)}
                for p in track.points
            ]
            for tid, track in tracks.items()
        },
    }


def tracks_from_json(data: dict):
    tracks = {}
    for tid_str, points in data["tracks"].items():
        tid = int(tid_str)
        track_points = [
            TrackPoint(frame=p["frame"], t=p["t"], bbox=tuple(p["bbox"]), centroid=tuple(p["centroid"]))
            for p in points
        ]
        tracks[tid] = Track(track_id=tid, points=track_points)
    return tracks, data["fps"], tuple(data["frame_size"])


def summarize_tracks(tracks: dict) -> list:
    """Human-readable summary to help a person pick which TID is the WR / CB."""
    rows = []
    for tid, track in tracks.items():
        if not track.points:
            continue
        n = len(track.points)
        avg_w = sum(p.bbox[2] - p.bbox[0] for p in track.points) / n
        avg_h = sum(p.bbox[3] - p.bbox[1] for p in track.points) / n
        rows.append({
            "track_id": tid,
            "frames_present": n,
            "first_frame": track.points[0].frame,
            "last_frame": track.points[-1].frame,
            "avg_box_w": round(avg_w, 1),
            "avg_box_h": round(avg_h, 1),
        })
    rows.sort(key=lambda r: -r["frames_present"])
    return rows
