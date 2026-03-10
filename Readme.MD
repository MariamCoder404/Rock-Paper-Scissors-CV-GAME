# ✊✋✌️ Rock · Paper · Scissors — Hand Gesture Game

A real-time Rock Paper Scissors game that uses your **webcam + hand gesture recognition** to detect your move. No external image files required — fully self-contained with a clean OpenCV UI.

---

## 📸 Demo

| State | Description |
|-------|-------------|
| Idle | Press `S` to start a round |
| Countdown | 3-second animated arc timer |
| Result | **WIN** 🟢 / **LOSE** 🔴 / **TIE** 🟡 shown instantly |

---

## 🚀 Features

- 🎥 Live webcam feed with real-time hand tracking
- ⏱️ 3-second animated countdown ring
- 🤖 AI opponent (random move each round)
- 📊 Live score tracking across multiple rounds
- 🎨 Custom dark UI — no PNG or resource files needed

---

## 🛠️ Requirements

- Python 3.8+
- A working webcam

### Dependencies

```bash
pip install opencv-python cvzone mediapipe numpy
```

| Package | Purpose |
|---------|---------|
| `opencv-python` | Video capture & UI rendering |
| `cvzone` | Hand detection wrapper |
| `mediapipe` | Underlying hand landmark model |
| `numpy` | Array operations |

---

## ▶️ Installation & Run

```bash
# 1. Clone the repo
git clone https://github.com/your-username/rps-game.git
cd rps-game

# 2. Install dependencies
pip install opencv-python cvzone mediapipe numpy

# 3. Run
python rps_game.py
```

---

## 🎮 How to Play

| Key | Action |
|-----|--------|
| `S` | Start a round / Next round |
| `Q` | Quit the game |

### Hand Gestures

| Move | Gesture | Fingers Up |
|------|---------|------------|
| ✊ Rock | Closed fist | `[0, 0, 0, 0, 0]` |
| ✋ Paper | Open hand | `[1, 1, 1, 1, 1]` |
| ✌️ Scissors | Index + middle up | `[0, 1, 1, 0, 0]` |

> **Tip:** Hold your hand clearly in front of the camera and keep your background simple for best detection accuracy.


## ⚙️ How It Works

1. Each frame from the webcam is scaled and cropped
2. `cvzone.HandDetector` locates hand landmarks and returns which fingers are extended
3. When `S` is pressed, a 3-second countdown starts
4. At the end of the timer, the player's gesture is read and the AI picks randomly (1–3)
5. Win logic is evaluated and the score updates

### Win Logic

```
Rock     beats  Scissors
Paper    beats  Rock
Scissors beats  Paper
```

---

## 🧠 AI Behavior

The AI is fully **random** — it calls `random.randint(1, 3)` each round with no memory or learning. Every round is a fair 1-in-3 chance regardless of history.

---

## 📄 License

MIT — free to use, modify, and distribute.