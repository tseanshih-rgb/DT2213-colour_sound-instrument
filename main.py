"""
Polyphonic colour-to-sound synthesizer — entry point.

Camera loop (main thread):
  1. Capture and mirror frame from FaceTime camera (index 0)
  2. Detect ALL colours present in the full frame above `threshold` % coverage
  3. Apply per-colour rolling-vote smoother (5-frame window)
  4. Push {colour: coverage_pct} to AudioEngine every frame
     → each channel's amplitude tracks its coverage percentage
  5. Draw stacked colour-swatch overlay + threshold HUD and display

Audio (sounddevice callback thread, managed by AudioEngine):
  • 8 always-running channels — one per colour
  • Channel amplitude = coverage_pct / 100 (proportional, not binary)
  • Amplitude transitions use the same 150 ms linear fade mechanism
  • Channels are summed and gain-normalised in real time

Runtime controls
----------------
  Q / ESC       quit
  UP arrow      raise detection threshold by 1 % (max 50 %)
  DOWN arrow    lower detection threshold by 1 % (min 1 %)

Arrow-key codes on macOS via cv2.waitKey:  UP = 63232, DOWN = 63233.
"""

import sys
from collections import deque

import cv2
import numpy as np

from colour_detector import detect_colours, COLOURS, MIN_COVERAGE_PCT
from audio_engine import AudioEngine

# ── display: BGR swatches ─────────────────────────────────────────────────────
_SWATCH_BGR: dict[str, tuple[int, int, int]] = {
    'red':    (0,   0,   220),
    'orange': (0,   140, 255),
    'yellow': (0,   255, 255),
    'green':  (0,   190, 0),
    'blue':   (200, 80,  0),
    'purple': (210, 0,   210),
    'white':  (255, 255, 255),
    'black':  (40,  40,  40),
}

_FONT = cv2.FONT_HERSHEY_SIMPLEX

# Overlay geometry constants
_ROW_H  = 44   # total height of one colour row (swatch + gap)
_SW_X0  = 12   # swatch left edge
_SW_W   = 38   # swatch width
_SW_GAP = 6    # vertical gap between rows (subtracted from swatch height)
_LBL_X  = 58   # label text left edge

# Threshold bounds for runtime adjustment
_THR_MIN = 1.0
_THR_MAX = 50.0

# macOS cv2.waitKey codes for arrow keys
_KEY_UP   = 63232
_KEY_DOWN = 63233


# ── per-colour rolling-vote smoother ─────────────────────────────────────────

SMOOTH_FRAMES = 5   # window length in frames (~167 ms at 30 fps)


class _Smoother:
    """
    Temporal smoother that applies an independent rolling vote to each colour.

    Each frame casts a boolean vote per colour (present / absent in the raw
    detection).  A colour is reported in the smoothed output only when it wins
    a strict majority (> 50 %) of votes in the most recent SMOOTH_FRAMES.

    Reported coverage = mean of the non-zero raw coverage values in the window,
    so the value reflects actual colour intensity rather than a time-diluted average.

    Pre-seeded with absent votes so early frames don't produce false positives.
    """

    def __init__(self) -> None:
        self._votes: dict[str, deque[bool]] = {
            c: deque([False] * SMOOTH_FRAMES, maxlen=SMOOTH_FRAMES) for c in COLOURS
        }
        self._covs: dict[str, deque[float]] = {
            c: deque([0.0] * SMOOTH_FRAMES, maxlen=SMOOTH_FRAMES) for c in COLOURS
        }

    def update(self, raw: dict[str, float]) -> dict[str, float]:
        """
        Push one raw detection frame; return smoothed {colour: avg_coverage_pct}.
        Colours absent from `raw` contribute a 0 vote this frame.
        """
        for c in COLOURS:
            self._votes[c].append(c in raw)
            self._covs[c].append(raw.get(c, 0.0))

        out: dict[str, float] = {}
        for c in COLOURS:
            # Strict majority: colour must be present in > half the recent window
            if sum(self._votes[c]) / SMOOTH_FRAMES > 0.5:
                non_zero = [v for v in self._covs[c] if v > 0.0]
                if non_zero:
                    out[c] = round(sum(non_zero) / len(non_zero), 1)
        return out


# ── overlay drawing ───────────────────────────────────────────────────────────

def _draw_overlay(
    frame: np.ndarray,
    detected: dict[str, float],
    threshold: float,
) -> None:
    """
    Draw the full HUD on `frame` in-place:

    Top-left  — vertical stack of colour swatches (one per detected colour,
                sorted descending by coverage), each labelled with name + pct.
    Bottom-left — threshold indicator, always visible regardless of detections.
    """
    frame_h = frame.shape[0]

    # ── Threshold HUD (bottom-left, always visible) ──────────────────────────
    # Displayed even when no colours are detected so the user can see the
    # current setting and adjust it.
    thr_label = f"Threshold: {threshold:.0f}%"
    thr_y     = frame_h - 14
    cv2.putText(frame, thr_label, (12, thr_y), _FONT, 0.60,
                (0, 0, 0), 2, cv2.LINE_AA)            # shadow
    cv2.putText(frame, thr_label, (12, thr_y), _FONT, 0.60,
                (220, 220, 220), 1, cv2.LINE_AA)       # foreground

    # ── Colour swatches (top-left) ────────────────────────────────────────────
    if not detected:
        cv2.putText(frame, "No colours detected",
                    (12, 42), _FONT, 0.75, (0, 0, 0),   3, cv2.LINE_AA)  # shadow
        cv2.putText(frame, "No colours detected",
                    (12, 42), _FONT, 0.75, (255, 255, 255), 1, cv2.LINE_AA)
        return

    # Sort by descending coverage so the dominant colour is at the top
    items = sorted(detected.items(), key=lambda kv: -kv[1])

    for row, (colour, pct) in enumerate(items):
        y0 = _SW_X0 + row * _ROW_H          # top of this swatch
        y1 = y0 + _ROW_H - _SW_GAP          # bottom of this swatch

        # Stop if the row would bleed over the threshold label
        if y1 + 4 > thr_y - 20:
            break

        bgr = _SWATCH_BGR.get(colour, (80, 80, 80))

        # Filled colour swatch + thin border
        cv2.rectangle(frame, (_SW_X0, y0), (_SW_X0 + _SW_W, y1), bgr,            -1)
        cv2.rectangle(frame, (_SW_X0, y0), (_SW_X0 + _SW_W, y1), (200, 200, 200), 1)

        # Label: shadow pass for legibility, then bright foreground pass
        label  = f"{colour.upper()}  {pct:.1f}%"
        text_y = y0 + (y1 - y0) // 2 + 6   # vertically centred in swatch row
        cv2.putText(frame, label, (_LBL_X, text_y), _FONT, 0.65,
                    (0, 0, 0),   3, cv2.LINE_AA)    # shadow
        cv2.putText(frame, label, (_LBL_X, text_y), _FONT, 0.65,
                    (255, 255, 255), 1, cv2.LINE_AA)


# ── entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    cap = cv2.VideoCapture(0)   # 0 = built-in FaceTime camera on macOS
    if not cap.isOpened():
        print("Error: cannot open camera index 0.", file=sys.stderr)
        print("Grant camera access in System Settings → Privacy → Camera.",
              file=sys.stderr)
        sys.exit(1)

    try:
        engine = AudioEngine()
        engine.start()
    except Exception as exc:
        print(f"Error starting audio engine: {exc}", file=sys.stderr)
        print("Make sure portaudio is installed:  brew install portaudio",
              file=sys.stderr)
        cap.release()
        sys.exit(1)

    smoother  = _Smoother()
    threshold = float(MIN_COVERAGE_PCT)   # start at the module default (15 %)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Warning: lost camera frame.", file=sys.stderr)
                continue

            # Horizontal flip for natural mirror / selfie orientation
            frame = cv2.flip(frame, 1)

            # Detect colours above the current runtime threshold
            raw      = detect_colours(frame, threshold)

            # Smooth per-colour across the last SMOOTH_FRAMES frames
            smoothed = smoother.update(raw)

            # Drive audio proportionally — coverage_pct becomes channel amplitude.
            # Called every frame so the 150 ms faders run continuously.
            engine.set_colours(smoothed)

            _draw_overlay(frame, smoothed, threshold)
            cv2.imshow("Colour Sound Synthesizer  [Q to quit]", frame)

            # ── key handling ─────────────────────────────────────────────────
            key = cv2.waitKey(1)
            if key == ord('q') or key == 27:      # Q or ESC → quit
                break
            elif key == _KEY_UP:                  # UP arrow → stricter threshold
                threshold = min(threshold + 1.0, _THR_MAX)
            elif key == _KEY_DOWN:                # DOWN arrow → more permissive
                threshold = max(threshold - 1.0, _THR_MIN)

    finally:
        cap.release()
        cv2.destroyAllWindows()
        engine.stop()


if __name__ == "__main__":
    main()
