# Colour Sound Instrument

A real-time colour-to-sound synthesizer that turns your camera into a musical instrument. Point it at any colour and hear it — red roars, blue breathes, purple drifts.

Built for **DT2213 DT2213 Musical Communication and Music Technology** at KTH Royal Institute of Technology.


## Demo Video

[![Watch Demo](https://img.youtube.com/vi/CG5qci6kjo8/maxresdefault.jpg)](https://youtu.be/CG5qci6kjo8)
---

## How It Works

The app reads your webcam feed and continuously detects which colours are visible on screen. Each colour has a unique synthesized sound designed to match its emotional character, not just a different pitch, but a different timbre, rhythm, and texture. Multiple colours can sound simultaneously, and louder means more of the frame is that colour.

| Colour | Sound Character |
|--------|----------------|
| 🔴 Red | Distorted sawtooth, fast tremolo — intense, dangerous |
| 🟠 Orange | Rhythmic pulse, warm square wave — energetic, upbeat |
| 🟡 Yellow | High-frequency staccato bursts — bright, joyful |
| 🟢 Green | Soft sine with slow breathing LFO — peaceful, natural |
| 🔵 Blue | Low sustained tones, reverb — deep, calm |
| 🟣 Purple | Detuned oscillators, pitch drift — dreamy, unstable |
| ⬜ White | Filtered white noise — vast, ethereal |
| ⬛ Black | Sub-bass rumble, heavy low-pass — dark, void |

If no colour is clearly detected, the output is completely silent.

---

## Requirements

- macOS
- Python 3.9+
- [Homebrew](https://brew.sh/)

---

## Installation

**1. Install PortAudio (required for audio output)**
```bash
brew install portaudio
```

**2. Clone the repository**
```bash
git clone https://github.com/tseanshih-rgb/DT2213-colour_sound-instrument.git
cd DT2213-colour_sound-instrument
```

**3. Install Python dependencies**
```bash
pip install -r requirements.txt
```

---

## Usage

```bash
python main.py
```

An OpenCV window will open showing your camera feed. Hold any coloured object up to the camera and listen.

**Controls**

| Key | Action |
|-----|--------|
| `↑` | Increase detection threshold by 1% |
| `↓` | Decrease detection threshold by 1% |
| `Q` | Quit |

**Detection threshold** controls how much of the frame a colour must occupy before it triggers a sound. The default is 15%. Increase it if your background is complex and causing unwanted sounds; decrease it if colours are not being picked up easily.

---

## Project Structure

```
colour_sound/
├── main.py              # Entry point: camera loop + colour detection
├── colour_detector.py   # HSV-based colour classification
├── synthesizer.py       # Per-colour synthesis functions
├── audio_engine.py      # sounddevice stream, fade + mixing logic
└── requirements.txt
```

---

## Dependencies

- [opencv-python](https://pypi.org/project/opencv-python/) — camera capture and display
- [sounddevice](https://python-sounddevice.readthedocs.io/) — real-time audio output
- [numpy](https://numpy.org/) — audio waveform synthesis

All audio is generated algorithmically. No audio samples or external sound files are used.

---

## Notes

- Colour detection uses HSV colour space, which is more robust to lighting variation than RGB
- All sounds fade in and out smoothly (150ms) when colours appear or disappear
- When multiple colours are detected simultaneously, their sounds mix together with loudness proportional to how much of the frame each colour occupies
- Colour detection works best under consistent, neutral lighting
