"""
HSV-based multi-colour classifier operating on the full camera frame.

OpenCV represents HSV as:  H 0-179  S 0-255  V 0-255
(Hue is halved from the conventional 0-360 range.)

Classification priority (applied in order, each pixel claimed at most once):
  1. Black  – low brightness regardless of hue / saturation
  2. White  – low saturation + high brightness
  3. Chromatic (Red, Orange, Yellow, Green, Blue, Purple) – need S ≥ 80
  4. Anything else → uncounted (ambiguous grey / low-sat colour)

`detect_colours` returns every colour whose pixel count reaches or exceeds
`threshold` (default MIN_COVERAGE_PCT = 15 %).  Multiple colours are returned
simultaneously when the scene contains several distinct regions.
"""

import cv2
import numpy as np

# Canonical tuple used by audio_engine and main as well — defines the
# universe of colours the whole system can reason about.
COLOURS: tuple[str, ...] = (
    'red', 'orange', 'yellow', 'green', 'blue', 'purple', 'white', 'black',
)

# Default minimum coverage a colour must reach to be reported.
# 15 % of a 640×480 frame ≈ 46 000 pixels — a region roughly 215×215 px.
# Adjustable at runtime via the UP / DOWN arrow keys in main.py.
MIN_COVERAGE_PCT: float = 15.0


def detect_colours(
    frame: np.ndarray,
    threshold: float = MIN_COVERAGE_PCT,
) -> dict[str, float]:
    """
    Classify every pixel in `frame` using HSV thresholds.

    Parameters
    ----------
    frame     : BGR image from cv2.VideoCapture.
    threshold : Minimum coverage percentage (0–100) a colour must reach
                to be included in the result.  Defaults to MIN_COVERAGE_PCT.

    Returns {colour_name: coverage_pct} for each qualifying colour.
    An empty dict means nothing clears the threshold.
    coverage_pct is a float in the range [threshold, 100].
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    H = hsv[:, :, 0].astype(np.int32)   # hue   0-179
    S = hsv[:, :, 1].astype(np.int32)   # sat   0-255
    V = hsv[:, :, 2].astype(np.int32)   # value 0-255
    total = H.size

    votes: dict[str, int] = {}

    # ── Black: low brightness regardless of hue or saturation ───────────────
    black = V < 50
    votes['black'] = int(black.sum())
    rest = ~black

    # ── White: desaturated and bright ───────────────────────────────────────
    # S < 40 keeps pale tints out; V > 180 accepts paper/walls, rejects mid-grey
    white = rest & (S < 40) & (V > 180)
    votes['white'] = int(white.sum())
    rest = rest & ~white

    # ── Chromatic colours: require S ≥ 80 to exclude washed-out greys ───────
    chroma = rest & (S >= 80)

    # Red wraps around 0/180 on the OpenCV hue wheel
    #   Low end  : H  0-10  (warm red)
    #   High end : H 165-179 (cool red / crimson)
    votes['red']    = int((chroma & ((H <= 10) | (H >= 165))).sum())

    # Orange: 11-25  (warm transitional band between red and yellow)
    votes['orange'] = int((chroma & (H >= 11) & (H <= 25)).sum())

    # Yellow: 26-34  (narrow bright-yellow band)
    votes['yellow'] = int((chroma & (H >= 26) & (H <= 34)).sum())

    # Green: 35-85  (wide natural band, includes yellow-green and teal)
    votes['green']  = int((chroma & (H >= 35) & (H <= 85)).sum())

    # Blue: 86-130  (cyan through mid-blue; stops before indigo)
    votes['blue']   = int((chroma & (H >= 86) & (H <= 130)).sum())

    # Purple: 131-164  (indigo through magenta, just before red wraps again)
    votes['purple'] = int((chroma & (H >= 131) & (H <= 164)).sum())

    result: dict[str, float] = {}
    for colour, count in votes.items():
        pct = count / total * 100.0
        if pct >= threshold:
            result[colour] = round(pct, 1)

    return result
