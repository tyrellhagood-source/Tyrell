"""Renders the annotated output video: bounding boxes with PID/TID labels,
a fading route-trace trail, and a small stat panel -- similar in spirit to
the "analyzing film with computer vision" reference clip this project is
modeled after.
"""
import subprocess

import cv2
import imageio_ffmpeg

from schema import PlayerIdentity, Track

ROLE_COLORS = {
    "WR": (60, 60, 235),   # red-ish (BGR)
    "CB": (235, 140, 60),  # blue-ish (BGR)
}
DEFAULT_COLOR = (0, 215, 255)
TRAIL_LEN = 25


def _color_for(identity: PlayerIdentity):
    return ROLE_COLORS.get(identity.role, DEFAULT_COLOR)


def render(video_path: str, out_path: str, tracks: dict, identities: dict,
           route_label: str = None, coverage_label: str = None):
    """tracks: dict[track_id] -> Track (raw tracker output)
    identities: dict[track_id] -> PlayerIdentity (human-assigned PID/role)
    """
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    ffmpeg_cmd = [
        ffmpeg_exe, "-y", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}", "-r", str(fps),
        "-i", "-",
        "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
        out_path,
    ]
    proc = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE)

    by_frame = {}
    for tid, track in tracks.items():
        for p in track.points:
            by_frame.setdefault(p.frame, []).append((tid, p))

    trails = {tid: [] for tid in tracks}

    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        for tid, point in by_frame.get(frame_idx, []):
            identity = identities.get(tid)
            color = _color_for(identity) if identity else DEFAULT_COLOR
            x1, y1, x2, y2 = [int(v) for v in point.bbox]
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            label = f"PID:{identity.name} TID:{tid}" if identity else f"TID:{tid}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 6, y1), color, -1)
            cv2.putText(frame, label, (x1 + 3, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

            trails[tid].append((int(point.centroid[0]), int(point.centroid[1])))
            if len(trails[tid]) > TRAIL_LEN:
                trails[tid].pop(0)
            for i in range(1, len(trails[tid])):
                cv2.line(frame, trails[tid][i - 1], trails[tid][i], color, 2)

        y = 30
        if route_label:
            cv2.putText(frame, f"Route: {route_label}", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
            y += 28
        if coverage_label:
            cv2.putText(frame, f"Coverage: {coverage_label}", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

        proc.stdin.write(frame.tobytes())
        frame_idx += 1

    cap.release()
    proc.stdin.close()
    proc.wait()
    return out_path
