"""
Per-colour audio synthesis functions.

Each public function has the signature:
    generate_<colour>(t: np.ndarray, sr: int) -> np.ndarray

where  t   is a 1-D array of sample times in seconds (continuous from
           stream start — not reset per buffer, so phase is always smooth)
       sr  is the sample rate in Hz
       return value is float32, amplitude roughly within ±1.

The dispatcher `generate(colour, t, sr)` routes to the right function.
"""

import numpy as np

π = np.pi  # shorthand used throughout


# ─────────────────────────────── helpers ────────────────────────────────────

def _sawtooth(freq: float, t: np.ndarray) -> np.ndarray:
    """Band-unlimited sawtooth ±1 — rich in harmonics, warm/harsh timbre."""
    return 2.0 * ((t * freq) % 1.0) - 1.0


def _lowpass(signal: np.ndarray, cutoff_hz: float, sr: int) -> np.ndarray:
    """
    First-order IIR low-pass (one-pole).

    y[n] = y[n-1] + α·(x[n] − y[n-1])

    α = ω_c / (ω_c + 1),  where ω_c = 2π·f_c / f_s

    Smaller cutoff → α closer to 0 → more aggressive filtering.
    Uses a Python loop over native floats (fast enough for ≤ 4096-sample
    buffers; avoids per-element numpy overhead).
    """
    omega = 2.0 * π * cutoff_hz / sr
    alpha = omega / (omega + 1.0)
    out = np.empty(len(signal), dtype=np.float64)
    y = float(signal[0])
    out[0] = y
    for i in range(1, len(signal)):
        y += alpha * (float(signal[i]) - y)
        out[i] = y
    return out


# ─────────────────────────────── Red ────────────────────────────────────────

def generate_red(t: np.ndarray, sr: int) -> np.ndarray:
    """
    Intense, dangerous.
    Sawtooth at A3 (220 Hz) → tanh waveshaper → 8 Hz tremolo.
    tanh(3·x) soft-clips into distortion without going hard-digital.
    """
    freq = 220.0  # A3 — aggressive mid-range
    wave = _sawtooth(freq, t)

    # Soft-clip / harmonic distortion: tanh(3x) keeps peak at ±1 but
    # crushes the waveform, adding odd harmonics (rough, buzzy).
    wave = np.tanh(3.0 * wave)

    # 8 Hz tremolo: fast amplitude flutter reinforces danger/urgency
    tremolo = 0.35 + 0.65 * np.sin(2.0 * π * 8.0 * t)

    return (wave * tremolo * 0.80).astype(np.float32)


# ────────────────────────────── Orange ──────────────────────────────────────

def generate_orange(t: np.ndarray, sr: int) -> np.ndarray:
    """
    Warm, energetic, rhythmic pulse.
    Sawtooth + second harmonic (warmth) at G3 (196 Hz),
    beat-synced amplitude envelope at 110 BPM.
    """
    freq = 196.0   # G3 — resonant, warm
    bpm  = 110.0
    beat = 60.0 / bpm   # 0.545 s per beat

    wave = _sawtooth(freq, t)
    # Second harmonic adds body without sharpness
    wave += 0.4 * np.sin(2.0 * π * freq * 2.0 * t)

    beat_phase = (t % beat) / beat          # 0→1 within each beat
    # Sharp attack at beat onset, exponential tail, silence in last 30 %
    # so the phase of the next beat's oscillator jump is inaudible.
    env = np.exp(-beat_phase * 4.0) * (beat_phase < 0.70).astype(float)

    return (wave * env * 0.55).astype(np.float32)


# ────────────────────────────── Yellow ──────────────────────────────────────

def generate_yellow(t: np.ndarray, sr: int) -> np.ndarray:
    """
    Happy, bouncy, delightful.
    Staccato major-pentatonic notes at 6 notes/sec in a high register.
    Notes: A5 C6 D6 E6 G6 (880–1568 Hz).

    Gate is closed for the last 50 % of each note slot so any phase
    discontinuity at a note change happens in silence (no click).
    """
    notes = np.array([880.0, 1046.5, 1174.7, 1318.5, 1568.0])  # A5 C6 D6 E6 G6
    note_dur = 1.0 / 6.0   # 6 notes per second

    idx   = (t / note_dur).astype(int) % len(notes)
    freqs = notes[idx]                          # per-sample frequency array

    phase_in_note = t % note_dur
    # 5 ms linear attack avoids click at note start
    attack = np.clip(phase_in_note / 0.005, 0.0, 1.0)
    # Gate closes at 50 % of note duration — silence before next note
    gate   = (phase_in_note < note_dur * 0.50).astype(float)
    env    = attack * gate

    wave = np.sin(2.0 * π * freqs * t)
    return (wave * env * 0.65).astype(np.float32)


# ─────────────────────────────── Green ──────────────────────────────────────

def generate_green(t: np.ndarray, sr: int) -> np.ndarray:
    """
    Peaceful, natural, woodwind-like.
    Additive harmonics approximate a flute/recorder: strong fundamental,
    progressively weaker upper partials.
    A slow 0.15 Hz "breathing" LFO creates inhale/exhale feel.
    """
    freq = 261.6  # C4 — middle C, gentle and centred

    # Harmonic series: bright enough to sound woodwind but not reedy/brassy.
    # Odd-harmonic weighting leans toward clarinet; even present for flute warmth.
    wave = (1.00 * np.sin(2.0 * π * freq * 1 * t) +
            0.30 * np.sin(2.0 * π * freq * 2 * t) +
            0.20 * np.sin(2.0 * π * freq * 3 * t) +
            0.10 * np.sin(2.0 * π * freq * 4 * t) +
            0.05 * np.sin(2.0 * π * freq * 5 * t))

    # Breathing LFO: 0.15 Hz ≈ one breath every 6.7 s
    lfo = 0.55 + 0.45 * np.sin(2.0 * π * 0.15 * t)

    return (wave * lfo * 0.38).astype(np.float32)


# ──────────────────────────────── Blue ──────────────────────────────────────

def generate_blue(t: np.ndarray, sr: int) -> np.ndarray:
    """
    Deep, calm, ocean-like.
    Sustained low fifth (C2 + G2) with pseudo-reverb: the dry signal is
    recomputed at multiple delayed time offsets and summed with decay.
    Because synthesis is deterministic from t, delayed copies are exact
    without needing a state buffer.
    """
    f1, f2 = 65.4, 98.0   # C2 and G2 — open fifth, stable and vast

    def _tone(t_arr: np.ndarray) -> np.ndarray:
        return 0.65 * np.sin(2.0 * π * f1 * t_arr) + 0.35 * np.sin(2.0 * π * f2 * t_arr)

    wet = _tone(t).astype(np.float64)

    # Reverb taps: (delay_s, gain) — longer delays fade further
    for delay_s, decay in [(0.05, 0.55), (0.13, 0.32), (0.25, 0.16), (0.42, 0.08)]:
        td   = t - delay_s
        mask = td >= 0.0
        if mask.any():
            echo       = np.zeros(len(t), dtype=np.float64)
            echo[mask] = _tone(td[mask])
            wet       += echo * decay

    return (wet * 0.28).astype(np.float32)


# ────────────────────────────── Purple ──────────────────────────────────────

def generate_purple(t: np.ndarray, sr: int) -> np.ndarray:
    """
    Dreamy, unstable.
    Two oscillators detuned by 8 cents create slow beating.
    Odd upper partials add complex, slightly dissonant overtones.
    An irregular LFO built from two incommensurate sines (0.7 and 1.37 Hz)
    produces a beating that never settles into a regular pattern.
    """
    freq       = 220.0                         # A3
    freq_sharp = freq * (2.0 ** (8.0 / 1200.0))  # 8 cents above A3

    # Detuned pair + overtone series (odd harmonics for otherworldliness)
    wave = (0.50 * np.sin(2.0 * π * freq       * t) +
            0.50 * np.sin(2.0 * π * freq_sharp  * t) +
            0.30 * np.sin(2.0 * π * freq * 2    * t) +
            0.20 * np.sin(2.0 * π * freq * 3    * t) +
            0.15 * np.sin(2.0 * π * freq * 5    * t) +
            0.10 * np.sin(2.0 * π * freq * 7    * t))
    # Theoretical peak amplitude = 1.75; normalise to ±1
    wave /= 1.75

    # Irregular amplitude envelope: 0.7 Hz and 1.37 Hz are incommensurate,
    # so the combined LFO never repeats on any short timescale.
    lfo = (0.50 +
           0.30 * np.sin(2.0 * π * 0.70  * t) +
           0.15 * np.sin(2.0 * π * 1.37  * t) +
           0.05 * np.sin(2.0 * π * 3.11  * t))
    # lfo range: 0.0 → 1.0 (fades to near-silence at LFO minima)

    return (wave * lfo * 0.50).astype(np.float32)


# ─────────────────────────────── White ──────────────────────────────────────

def generate_white(t: np.ndarray, sr: int) -> np.ndarray:
    """
    Pure, vast.
    White noise at low amplitude — no tonal content, completely flat spectrum.
    """
    noise = np.random.randn(len(t))
    return (noise * 0.12).astype(np.float32)


# ─────────────────────────────── Black ──────────────────────────────────────

def generate_black(t: np.ndarray, sr: int) -> np.ndarray:
    """
    Dark, void.
    40 Hz sub-bass oscillator with second partial, reverb via delayed copies,
    then heavy low-pass (100 Hz cutoff) to remove any midrange presence.
    """
    freq = 40.0   # sub-bass — felt more than heard

    def _osc(t_arr: np.ndarray) -> np.ndarray:
        return (np.sin(2.0 * π * freq       * t_arr) +
                0.35 * np.sin(2.0 * π * freq * 2 * t_arr))

    wet = _osc(t).astype(np.float64)

    # Reverb taps — longer than Blue's to emphasise the void-like space
    for delay_s, decay in [(0.08, 0.55), (0.20, 0.32), (0.38, 0.18)]:
        td   = t - delay_s
        mask = td >= 0.0
        if mask.any():
            echo       = np.zeros(len(t), dtype=np.float64)
            echo[mask] = _osc(td[mask])
            wet       += echo * decay

    # Low-pass at 100 Hz: α ≈ 0.014 at 44100 Hz → very heavy roll-off
    filtered = _lowpass(wet, cutoff_hz=100.0, sr=sr)
    return (filtered * 0.28).astype(np.float32)


# ─────────────────────────── dispatcher ─────────────────────────────────────

_SYNTHS = {
    'red':    generate_red,
    'orange': generate_orange,
    'yellow': generate_yellow,
    'green':  generate_green,
    'blue':   generate_blue,
    'purple': generate_purple,
    'white':  generate_white,
    'black':  generate_black,
}


def generate(colour: str | None, t: np.ndarray, sr: int) -> np.ndarray:
    """Dispatch to the correct synth; return zeros for unknown/None colours."""
    fn = _SYNTHS.get(colour)  # type: ignore[arg-type]
    if fn is None:
        return np.zeros(len(t), dtype=np.float32)
    return fn(t, sr)
