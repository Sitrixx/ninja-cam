"""
Ninja Cam - make a peace sign and vanish in a puff of smoke.

Reads your webcam, detects a peace sign, and sends the result to a virtual
camera (OBS Virtual Camera) so Teams/Zoom/Meet can use it as a webcam.

Keys (in the preview window):
  B      capture a clean background plate (step out of frame during countdown)
  SPACE  toggle vanish manually
  M      mirror preview on/off
  Q/ESC  quit
"""

import argparse
import math
import os
import random
import sys
import time
import urllib.request

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision

HERE = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(HERE, "models")
MODELS = {
    "hand_landmarker.task": "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task",
    "selfie_segmenter.tflite": "https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_segmenter/float16/latest/selfie_segmenter.tflite",
}

# Processing resolution for the ML models (output stays full resolution).
PROC_W, PROC_H = 640, 360

PEACE_HOLD_SEC = 0.45   # how long the sign must be held to trigger
COOLDOWN_SEC = 2.0      # ignore gestures for this long after a trigger
PUFF_SEC = 1.6          # length of the smoke animation
SWAP_AT = 0.22          # fraction of the puff at which you (dis)appear


def ensure_models():
    os.makedirs(MODELS_DIR, exist_ok=True)
    for name, url in MODELS.items():
        path = os.path.join(MODELS_DIR, name)
        if not os.path.exists(path):
            print(f"Downloading {name} ...")
            urllib.request.urlretrieve(url, path)
    return {name: os.path.join(MODELS_DIR, name) for name in MODELS}


# ---------------------------------------------------------------- gesture ---

def _dist(a, b):
    return math.hypot(a.x - b.x, a.y - b.y)


def is_peace_sign(lm):
    """lm: 21 normalized hand landmarks. Index + middle up and apart, ring + pinky curled."""
    wrist = lm[0]
    palm = _dist(wrist, lm[9]) + 1e-6  # wrist -> middle MCP, used as scale

    def extended(tip, pip):
        return _dist(wrist, lm[tip]) > _dist(wrist, lm[pip]) * 1.15

    def curled(tip, pip):
        return _dist(wrist, lm[tip]) < _dist(wrist, lm[pip]) * 1.05

    if not (extended(8, 6) and extended(12, 10)):
        return False
    if not (curled(16, 14) and curled(20, 18)):
        return False
    # Fingers must be spread into a V, not held together.
    return _dist(lm[8], lm[12]) / palm > 0.35


# ------------------------------------------------------------------ smoke ---

def _make_sprites(n=8, size=96):
    """Lumpy puff sprites as (alpha, light) pairs, lit from above."""
    yy, xx = np.mgrid[-1:1:size * 1j, -1:1:size * 1j].astype(np.float32)
    rng = np.random.default_rng(7)
    sprites = []
    for _ in range(n):
        noise = np.zeros((size, size), np.float32)
        for octave, w in ((4, 0.6), (8, 0.3), (16, 0.1)):
            o = rng.random((octave, octave)).astype(np.float32)
            noise += w * cv2.resize(o, (size, size), interpolation=cv2.INTER_CUBIC)
        r = np.sqrt(xx ** 2 + yy ** 2) + (noise - 0.5) * 0.55
        alpha = np.clip((1.0 - r) * 2.2, 0, 1) ** 1.5
        light = np.clip(0.82 - 0.22 * yy + 0.25 * (noise - 0.5), 0.45, 1.0)
        sprites.append((alpha.astype(np.float32), light.astype(np.float32)))
    return sprites


SMOKE_DOWNSCALE = 4  # smoke is soft, so it's rendered at 1/4 resolution
SMOKE_BGR = (238, 240, 242)


class SmokePuff:
    def __init__(self, cx, cy, scale, frame_w, frame_h):
        """cx, cy: puff center; scale: roughly the size of the person in pixels."""
        self.t0 = time.time()
        self.w, self.h = frame_w, frame_h
        d = SMOKE_DOWNSCALE
        self.sw, self.sh = frame_w // d, frame_h // d
        cx, cy, scale = cx / d, cy / d, scale / d
        self.sprites = _make_sprites()
        self.particles = []
        for i in range(90):
            ang = random.uniform(0, 2 * math.pi)
            speed = random.uniform(0.15, 0.6) * scale
            self.particles.append({
                "x": cx + random.gauss(0, scale * 0.18),
                "y": cy + random.gauss(0, scale * 0.25),
                "vx": math.cos(ang) * speed,
                "vy": math.sin(ang) * speed * 0.7 - scale * 0.12,  # drift up
                "r0": random.uniform(0.07, 0.15) * scale,
                "grow": random.uniform(0.12, 0.28) * scale,
                "sprite": random.randrange(len(self.sprites)),
                "delay": random.uniform(0, 0.1),
                "life": random.uniform(0.7, 1.0) * PUFF_SEC,
                "rise": random.uniform(5, 15),
            })
        # Big ones first so smaller puffs layer on top.
        self.particles.sort(key=lambda p: -p["grow"])

    def progress(self):
        return (time.time() - self.t0) / PUFF_SEC

    def done(self):
        return self.progress() >= 1.0

    def render(self, frame):
        """Return frame with the smoke composited on top."""
        t_now = time.time() - self.t0
        alpha = np.zeros((self.sh, self.sw), np.float32)
        light = np.zeros((self.sh, self.sw), np.float32)
        for p in self.particles:
            t = t_now - p["delay"]
            if t <= 0 or t >= p["life"]:
                continue
            k = t / p["life"]
            ease = 1 - (1 - min(1.0, t / 0.45)) ** 3  # fast burst, then slow drift
            x = p["x"] + p["vx"] * ease
            y = p["y"] + p["vy"] * ease - p["rise"] * t
            r = int(p["r0"] + p["grow"] * ease)
            if r < 1:
                continue
            a = min(1.0, t / 0.05) * (1 - k ** 2) * 0.9
            spr_a, spr_l = self.sprites[p["sprite"]]
            spr_a = cv2.resize(spr_a, (2 * r, 2 * r))
            spr_l = cv2.resize(spr_l, (2 * r, 2 * r))
            x0, y0 = int(x) - r, int(y) - r
            cx0, cy0 = max(x0, 0), max(y0, 0)
            cx1, cy1 = min(x0 + 2 * r, self.sw), min(y0 + 2 * r, self.sh)
            if cx0 >= cx1 or cy0 >= cy1:
                continue
            sa = spr_a[cy0 - y0:cy1 - y0, cx0 - x0:cx1 - x0] * a
            sl = spr_l[cy0 - y0:cy1 - y0, cx0 - x0:cx1 - x0]
            # "over" compositing so puffs layer instead of washing out
            al = alpha[cy0:cy1, cx0:cx1]
            light[cy0:cy1, cx0:cx1] = light[cy0:cy1, cx0:cx1] * (1 - sa) + sl * sa
            alpha[cy0:cy1, cx0:cx1] = al + sa * (1 - al)
        ys, xs = np.nonzero(alpha > 0.004)
        if not len(xs):
            return frame
        # Only composite inside the smoke's bounding box.
        d = SMOKE_DOWNSCALE
        x0, x1, y0, y1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
        alpha, light = alpha[y0:y1, x0:x1], light[y0:y1, x0:x1]
        shade = light / np.maximum(alpha, 1e-4)
        X0, Y0 = x0 * d, y0 * d
        X1, Y1 = min(x1 * d, self.w), min(y1 * d, self.h)
        size = (X1 - X0, Y1 - Y0)
        a = cv2.resize(alpha, size, interpolation=cv2.INTER_LINEAR)
        shade = cv2.resize(shade, size, interpolation=cv2.INTER_LINEAR)
        color = cv2.merge([cv2.convertScaleAbs(shade, alpha=float(c)) for c in SMOKE_BGR])
        out = frame.copy()
        out[Y0:Y1, X0:X1] = cv2.blendLinear(out[Y0:Y1, X0:X1], color, 1 - a, a)
        return out


# ------------------------------------------------------------- background ---

class BackgroundPlate:
    """Learns what's behind you from pixels where you are not standing."""

    def __init__(self, h, w):
        self.plate = None
        self.seen = np.zeros((h, w), np.float32)  # 1 where we've seen the background
        self.margin = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))

    def update(self, frame, person_small, rate=0.08):
        if self.plate is None:
            self.plate = frame.astype(np.float32)
        h, w = self.seen.shape
        # Keep a safety margin around the person: edge pixels are often a mix of you + background.
        bg_small = cv2.erode((person_small < 0.15).astype(np.uint8), self.margin)
        bg = cv2.resize(bg_small, (w, h), interpolation=cv2.INTER_NEAREST) > 0
        first_time = bg & (self.seen == 0)
        self.plate[first_time] = frame[first_time]  # never seen before: take it as-is
        cv2.accumulateWeighted(frame, self.plate, rate, mask=bg.astype(np.uint8))
        self.seen[bg] = 1.0

    def capture_clean(self, frames):
        self.plate = np.median(np.stack(frames), axis=0).astype(np.float32)
        self.seen[:] = 1.0

    def filled(self):
        """Plate with never-seen areas (behind you) filled in by inpainting."""
        plate = np.clip(self.plate, 0, 255).astype(np.uint8)
        hole = (self.seen < 0.5).astype(np.uint8) * 255
        if hole.any():
            h, w = hole.shape
            small = cv2.resize(plate, (w // 4, h // 4), interpolation=cv2.INTER_AREA)
            small_hole = cv2.resize(hole, (w // 4, h // 4), interpolation=cv2.INTER_NEAREST)
            small_hole = cv2.dilate(small_hole, self.margin)  # don't trust pixels bordering the hole
            small = cv2.inpaint(small, small_hole, 5, cv2.INPAINT_TELEA)
            big = cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC)
            m = cv2.GaussianBlur(hole.astype(np.float32) / 255, (0, 0), 6)[..., None]
            plate = (plate * (1 - m) + big * m).astype(np.uint8)
        return plate


# ------------------------------------------------------------------- main ---

def open_camera(index, w, h):
    backend = cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY
    cap = cv2.VideoCapture(index, backend)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
    cap.set(cv2.CAP_PROP_FPS, 30)
    if not cap.isOpened():
        sys.exit(f"Could not open camera {index}. Try --camera 1, and close Teams' preview of the real webcam.")
    return cap


def open_virtual_cam(w, h, fps):
    try:
        import pyvirtualcam
        cam = pyvirtualcam.Camera(width=w, height=h, fps=fps, fmt=pyvirtualcam.PixelFormat.BGR)
        print(f"Virtual camera ready: {cam.device}  -> pick it as your camera in Teams")
        return cam
    except Exception as e:
        print(f"[!] No virtual camera ({e}).\n    Install OBS Studio (it provides 'OBS Virtual Camera'). Running preview only.")
        return None


def draw_hud(img, lines):
    y = 28
    for text in lines:
        cv2.putText(img, text, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(img, text, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        y += 26


def main():
    ap = argparse.ArgumentParser(description="Peace sign -> vanish in smoke, as a virtual webcam.")
    ap.add_argument("--camera", type=int, default=0, help="webcam index (default 0)")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--no-preview", action="store_true", help="don't show the preview window")
    args = ap.parse_args()

    paths = ensure_models()
    hands = vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
        base_options=mp_tasks.BaseOptions(model_asset_path=paths["hand_landmarker.task"]),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.6,
        min_tracking_confidence=0.5,
    ))
    segmenter = vision.ImageSegmenter.create_from_options(vision.ImageSegmenterOptions(
        base_options=mp_tasks.BaseOptions(model_asset_path=paths["selfie_segmenter.tflite"]),
        running_mode=vision.RunningMode.VIDEO,
        output_confidence_masks=True,
        output_category_mask=False,
    ))

    cap = open_camera(args.camera, args.width, args.height)
    ok, frame = cap.read()
    if not ok:
        sys.exit("Camera returned no frames.")
    H, W = frame.shape[:2]
    vcam = open_virtual_cam(W, H, args.fps)

    plate = BackgroundPlate(H, W)
    vanished = False
    hidden_plate = None       # background used while vanished
    puff = None
    swapped = True            # whether the current puff already flipped visibility
    peace_since = None
    last_trigger = 0.0
    capture_until = None      # clean-plate capture state
    capture_frames = []
    force_toggle = False      # SPACE pressed
    mirror = True
    t_start = time.time()
    dilate_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13))
    fps_t, fps_n, fps = time.time(), 0, 0.0

    print("Ready! Hold a peace sign to vanish, do it again to come back.")
    print("Tip: press B and step out of frame for 5s first - it makes the empty chair look perfect.")

    while True:
        ok, frame = cap.read()
        if not ok:
            continue
        now = time.time()
        ts = int((now - t_start) * 1000)

        small = cv2.resize(frame, (PROC_W, PROC_H))
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(small, cv2.COLOR_BGR2RGB))

        seg = segmenter.segment_for_video(mp_img, ts)
        person_small = seg.confidence_masks[0].numpy_view().astype(np.float32)
        person_small = np.squeeze(person_small)

        # --- gesture ---
        res = hands.detect_for_video(mp_img, ts)
        peace = any(is_peace_sign(lm) for lm in res.hand_landmarks)
        if peace:
            peace_since = peace_since or now
        else:
            peace_since = None
        held = peace_since is not None and now - peace_since >= PEACE_HOLD_SEC

        # --- clean plate capture (B) ---
        if capture_until is not None:
            if now >= capture_until:
                capture_frames.append(frame.copy())
                if len(capture_frames) >= 20:
                    plate.capture_clean(capture_frames)
                    capture_until, capture_frames = None, []
                    print("Clean background captured.")
        elif not vanished:
            plate.update(frame, person_small)

        # --- trigger ---
        trigger = (held and now - last_trigger > COOLDOWN_SEC or force_toggle) and puff is None
        force_toggle = False
        if trigger:
            last_trigger, peace_since = now, None
            ys, xs = np.nonzero(person_small > 0.5)
            if len(xs):
                cx, cy = xs.mean() * W / PROC_W, ys.mean() * H / PROC_H
                scale = max(np.ptp(xs) * W / PROC_W, np.ptp(ys) * H / PROC_H) * 0.6
            else:
                cx, cy, scale = W / 2, H / 2, H * 0.5
            puff = SmokePuff(cx, cy, max(scale, H * 0.3), W, H)
            swapped = False
            if not vanished:
                hidden_plate = plate.filled()

        if puff is not None and not swapped and puff.progress() >= SWAP_AT:
            vanished = not vanished
            swapped = True

        # --- composite ---
        out = frame
        if vanished:
            # Grow + feather the person mask (at low res, it's cheaper) to hide edges.
            m = cv2.dilate((person_small > 0.3).astype(np.uint8), dilate_k)
            m = cv2.GaussianBlur(m.astype(np.float32), (0, 0), 3.5)
            m = cv2.resize(m, (W, H))
            out = cv2.blendLinear(frame, hidden_plate, 1 - m, m)
        if puff is not None:
            out = puff.render(out)
            if puff.done():
                puff = None

        if vcam is not None:
            vcam.send(out)

        # --- preview ---
        fps_n += 1
        if now - fps_t >= 1:
            fps, fps_n, fps_t = fps_n / (now - fps_t), 0, now
        if not args.no_preview:
            prev = cv2.flip(out, 1) if mirror else out.copy()
            hud = [f"{'VANISHED' if vanished else 'visible'}   {fps:.0f} fps   "
                   f"{'-> virtual cam' if vcam else 'NO virtual cam'}"]
            if capture_until is not None:
                left = capture_until - now
                hud.append(f"Step out of frame! capturing in {left:.1f}s" if left > 0 else "capturing...")
            elif peace_since:
                hud.append("peace sign detected... hold it")
            else:
                hud.append("peace sign = vanish/return | B clean bg | SPACE toggle | Q quit")
            draw_hud(prev, hud)
            cv2.imshow("Ninja Cam", prev)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            elif key == ord("b") and capture_until is None and not vanished:
                capture_until, capture_frames = now + 5.0, []
            elif key == ord(" "):
                force_toggle = True
            elif key == ord("m"):
                mirror = not mirror

    cap.release()
    if vcam is not None:
        vcam.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
