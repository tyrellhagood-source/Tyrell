"""Experimental jersey-number OCR, as an assist to (not a replacement for)
the manual PID tagging in identity.py.

Why this isn't the default identification path: broadcast/phone football
footage gives you a small, motion-blurred, often partially-occluded digit
region, at an oblique angle, for a fraction of a second at a time. Every
published jersey-number recognition system that actually works well (e.g.
the SoccerNet jersey-number challenge winners) does it with a dedicated
trained pipeline -- a legibility classifier to throw out unreadable frames,
a number-specific recognition model, and temporal voting across a whole
track -- fine-tuned on that sport's jersey style. A generic off-the-shelf
OCR engine run straight on a crop, like this module, will misread digits
constantly (6/8/9, 1/7, etc.) and should be treated as a hint to speed up
manual tagging, never as ground truth.

If this turns out to matter enough for the app, the right next step is
fine-tuning something like mkoshkina/jersey-number-pipeline on football
jerseys rather than hardening this module further.
"""
from collections import Counter

import cv2

from schema import Track

_reader = None


def _get_reader():
    global _reader
    if _reader is None:
        import easyocr
        _reader = easyocr.Reader(["en"], gpu=False, verbose=False)
    return _reader


def _torso_crop(frame, bbox):
    x1, y1, x2, y2 = [int(v) for v in bbox]
    h = y2 - y1
    # jersey numbers sit roughly in the chest/back third of the bbox, not the
    # legs or helmet -- narrowing the crop cuts down on false reads a lot.
    ty1 = y1 + int(h * 0.12)
    ty2 = y1 + int(h * 0.55)
    return frame[max(ty1, 0):max(ty2, 0), max(x1, 0):max(x2, 0)]


def _sample_points(track: Track, n: int):
    pts = track.points
    if len(pts) <= n:
        return pts
    step = len(pts) / n
    return [pts[int(i * step)] for i in range(n)]


def suggest_jersey_numbers(video_path: str, tracks: dict, sample_frames: int = 8) -> dict:
    """Returns dict[track_id] -> (number_str, confidence) for tracks where at
    least one frame produced a plausible 1-2 digit read. Tracks with no
    readable number are omitted entirely (not guessed)."""
    reader = _get_reader()
    cap = cv2.VideoCapture(video_path)
    results = {}

    for tid, track in tracks.items():
        votes = Counter()
        conf_sum = Counter()
        for point in _sample_points(track, sample_frames):
            cap.set(cv2.CAP_PROP_POS_FRAMES, point.frame)
            ok, frame = cap.read()
            if not ok:
                continue
            crop = _torso_crop(frame, point.bbox)
            if crop.size == 0:
                continue
            try:
                reads = reader.readtext(crop, allowlist="0123456789", detail=1)
            except Exception:
                continue
            for _, text, conf in reads:
                text = text.strip()
                if text.isdigit() and 1 <= len(text) <= 2 and conf >= 0.35:
                    votes[text] += 1
                    conf_sum[text] += conf

        if votes:
            best, count = votes.most_common(1)[0]
            avg_conf = conf_sum[best] / count
            # discount confidence when only one of several sampled frames agreed
            agreement = count / sum(votes.values())
            results[tid] = (best, round(avg_conf * agreement, 2))

    cap.release()
    return results
