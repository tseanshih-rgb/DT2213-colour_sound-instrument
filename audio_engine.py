"""
Polyphonic real-time audio engine backed by a sounddevice OutputStream.

Architecture
------------
Eight synthesizer channels run in parallel, one per colour.  Each channel
has an independent gain fader that moves linearly toward its target:

  colour at C % coverage  →  target = C / 100   →  fade toward C/100 amplitude
  colour absent           →  target = 0.0        →  fade OUT over FADE_LEN samples

Channel amplitude is therefore proportional to coverage: a colour filling
50 % of the frame drives its channel at 0.5 amplitude; 100 % gives full
amplitude; the minimum-threshold colour (e.g. 15 %) plays at 0.15.

The callback generates samples for every channel whose gain is non-zero,
sums them (additive mixing), then normalises the mix so the combined gain
never exceeds 1.0 in expectation.  A hard clip is applied as a safety net.

Thread-safety
-------------
Two brief lock acquisitions per callback:
  1. Snapshot current_gains / target_gains / synth_time (read-only).
  2. Write back updated gains and advance synth_time.
All synthesis runs outside the lock so the audio thread is never blocked
while the main thread updates target gains.
"""

import threading

import numpy as np
import sounddevice as sd

from colour_detector import COLOURS
from synthesizer import generate

SAMPLE_RATE = 44_100                     # Hz
BLOCK_SIZE  = 1_024                      # frames per callback ≈ 23 ms
FADE_LEN    = int(0.15 * SAMPLE_RATE)    # 150 ms linear fade = 6 615 samples

# Gains below this are treated as zero — avoids synthesising for channels
# that have faded to numerical noise rather than exactly 0.0.
_GAIN_FLOOR = 1e-9


class AudioEngine:
    def __init__(self) -> None:
        self._lock = threading.Lock()

        # Monotonically increasing playback clock (seconds).
        # All 8 channels share this clock so their oscillator phases are
        # globally coherent — fading a channel in sounds like unmuting it,
        # not like starting a new oscillator from phase 0.
        self._synth_time: float = 0.0

        # Per-channel independent gain state (values always in [0.0, 1.0])
        self._current_gains: dict[str, float] = {c: 0.0 for c in COLOURS}
        self._target_gains:  dict[str, float] = {c: 0.0 for c in COLOURS}

        self._stream = sd.OutputStream(
            samplerate=SAMPLE_RATE,
            channels=2,
            dtype='float32',
            blocksize=BLOCK_SIZE,
            callback=self._callback,
        )

    # ── public API ───────────────────────────────────────────────────────────

    def start(self) -> None:
        self._stream.start()

    def stop(self) -> None:
        self._stream.stop()
        self._stream.close()

    def set_colours(self, detected: dict[str, float]) -> None:
        """
        Called from the main thread once per camera frame.
        `detected` maps colour_name → coverage_pct (from the smoother).

        Target gain = coverage_pct / 100.0  (so 50 % coverage → gain 0.5).
        Absent colours get target 0.0.  The fader moves at a fixed rate
        (1 / FADE_LEN per sample) so partial transitions complete faster than
        full 0→1 ones, but always use the same smooth ramp mechanism.
        """
        with self._lock:
            for colour in COLOURS:
                self._target_gains[colour] = detected.get(colour, 0.0) / 100.0

    # ── sounddevice callback (audio thread) ──────────────────────────────────

    def _callback(
        self,
        outdata: np.ndarray,
        frames: int,
        time_info,
        status,
    ) -> None:
        if status:
            print(f"[audio] {status}", flush=True)

        # ── 1. Snapshot mutable state (brief lock, no synthesis inside) ──────
        with self._lock:
            current_gains = dict(self._current_gains)
            target_gains  = dict(self._target_gains)
            t0            = self._synth_time

        # ── 2. Build shared time array for all channels this buffer ──────────
        # endpoint=False: the next buffer starts exactly where this one ended,
        # keeping every oscillator phase perfectly continuous.
        t = np.linspace(t0, t0 + frames / SAMPLE_RATE, frames, endpoint=False)

        # ── 3. Polyphonic mix ────────────────────────────────────────────────
        mix        = np.zeros(frames, dtype=np.float64)
        total_gain = np.zeros(frames, dtype=np.float64)
        new_gains: dict[str, float] = {}

        # Maximum gain change allowed in this buffer: moves at rate 1/FADE_LEN
        max_delta = frames / FADE_LEN

        for colour in COLOURS:
            current = current_gains[colour]
            target  = target_gains[colour]

            # Advance current gain toward target by at most max_delta
            raw_delta = target - current
            end = current + float(np.clip(raw_delta, -max_delta, max_delta))

            # Snap to exact target to prevent floating-point drift accumulating
            # (e.g. a channel that should be silent stays at exactly 0.0)
            if abs(end - target) < 1e-9:
                end = target

            new_gains[colour] = end

            # Skip synthesis entirely for channels that are and remain silent
            if current < _GAIN_FLOOR and end < _GAIN_FLOOR:
                continue

            # Smooth per-sample gain ramp: starts at `current`, ends at `end`.
            # Using endpoint=True means the last sample reaches `end` exactly,
            # and the next buffer begins its linspace from `end` → no step.
            gain_arr = np.linspace(current, end, frames)   # float64 by default

            samples = generate(colour, t, SAMPLE_RATE).astype(np.float64)
            mix        += samples * gain_arr
            total_gain += gain_arr

        # ── 4. Normalise: keep summed gain ≤ 1.0 per sample ─────────────────
        # When one channel is at gain 1.0:  total_gain = 1 → no change.
        # When N channels are all fully active: each contributes 1/N amplitude.
        # This prevents hard clipping when many colours are active at once.
        norm = np.maximum(total_gain, 1.0)
        mix /= norm

        # ── 5. Commit gain state (brief lock) ────────────────────────────────
        with self._lock:
            self._synth_time += frames / SAMPLE_RATE
            self._current_gains.update(new_gains)

        # ── 6. Write stereo output with hard-clip safety net ─────────────────
        clipped       = np.clip(mix, -1.0, 1.0).astype(np.float32)
        outdata[:, 0] = clipped   # left
        outdata[:, 1] = clipped   # right (same mono signal)
