import random
import cv2
import cvzone
from cvzone.HandTrackingModule import HandDetector
import time
import numpy as np

cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

detector = HandDetector(maxHands=1)

#Game State 
timer        = 0
stateResult  = False
startGame    = False
initialTime  = 0
score        = [0, 0]   # [AI, Player]
playerMove   = None
aiMove       = None
roundResult  = ""        # "WIN" / "LOSE" / "TIE"
resultTimer  = 0

# Palette 
BG_DARK   = (18,  18,  30)
BG_MID    = (28,  28,  48)
ACCENT1   = (80, 200, 255)   # cyan
ACCENT2   = (255, 80, 160)   # hot-pink
GOLD      = (50, 210, 255)   # amber (BGR)
WHITE     = (240, 240, 255)
GREY      = (120, 120, 150)
WIN_COL   = (80, 230, 120)
LOSE_COL  = (80,  80, 230)
TIE_COL   = (200, 200, 80)

CANVAS_W, CANVAS_H = 1280, 720



# Drawing helpers

def draw_gradient_bg(img):
    """Vertical gradient background."""
    for y in range(CANVAS_H):
        t = y / CANVAS_H
        b = int(18  + t * 10)
        g = int(18  + t * 8)
        r = int(30  + t * 20)
        img[y, :] = (b, g, r)

def draw_rounded_rect(img, x1, y1, x2, y2, color, radius=20, thickness=-1, alpha=1.0):
    overlay = img.copy()
    cv2.rectangle(overlay, (x1+radius, y1), (x2-radius, y2), color, thickness)
    cv2.rectangle(overlay, (x1, y1+radius), (x2, y2-radius), color, thickness)
    for cx, cy in [(x1+radius, y1+radius),(x2-radius, y1+radius),
                   (x1+radius, y2-radius),(x2-radius, y2-radius)]:
        cv2.circle(overlay, (cx, cy), radius, color, thickness)
    cv2.addWeighted(overlay, alpha, img, 1-alpha, 0, img)


def put_text_centered(img, text, cx, cy, font, scale, color, thickness):
    (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
    cv2.putText(img, text, (cx - tw//2, cy + th//2), font, scale, color, thickness, cv2.LINE_AA)

def put_text_shadow(img, text, x, y, font, scale, color, thickness):
    cv2.putText(img, text, (x+2, y+2), font, scale, (0,0,0), thickness+1, cv2.LINE_AA)
    cv2.putText(img, text, (x, y), font, scale, color, thickness, cv2.LINE_AA)


# Move display — simple text labels + hand pose strings
MOVE_POSE  = {1: "Rock",     2: "Paper",  3: "Scissors", None: "?"}
MOVE_NAMES = {1: "ROCK",     2: "PAPER",  3: "SCISSORS", None: "?"}
MOVE_COLOR = {1: ACCENT1,    2: ACCENT2,  3: GOLD,       None: GREY}


# Panel drawing


def draw_panel(img, x1, y1, x2, y2, title, move_id, score_val, is_player=False):
    """Draw a side panel (AI or Player)."""
    # background card
    draw_rounded_rect(img, x1, y1, x2, y2, BG_MID, radius=18, alpha=0.85)
    # border glow
    col = MOVE_COLOR.get(move_id, GREY)
    draw_rounded_rect(img, x1, y1, x2, y2, col, radius=18, thickness=2, alpha=0.6)

    cx = (x1+x2)//2
    # title
    put_text_shadow(img, title, cx - 55, y1+42,
                    cv2.FONT_HERSHEY_DUPLEX, 0.9, WHITE, 2)

    # score bubble
    cv2.circle(img, (cx, y1+105), 45, col, -1)
    cv2.circle(img, (cx, y1+105), 45, WHITE, 2)
    put_text_centered(img, str(score_val), cx, y1+105,
                      cv2.FONT_HERSHEY_DUPLEX, 1.8, BG_DARK, 3)

    # move display — big pose text + name label
    icon_cy = (y1+y2)//2 + 30
    pose = MOVE_POSE.get(move_id, "?")
    put_text_centered(img, pose, cx, icon_cy,
                      cv2.FONT_HERSHEY_DUPLEX, 2.2, col, 3)

    # move label
    put_text_centered(img, MOVE_NAMES.get(move_id, "?"),
                      cx, y2-30, cv2.FONT_HERSHEY_PLAIN, 1.6, col, 2)


def draw_center_zone(img, t_remaining, state_result, round_result):
    cx = CANVAS_W // 2
    # VS divider
    cv2.line(img, (cx, 200), (cx, CANVAS_H-80), (60,60,90), 2)

    if not startGame:
        put_text_centered(img, "PRESS  [S]  TO  START",
                          cx, CANVAS_H//2, cv2.FONT_HERSHEY_DUPLEX, 0.9, ACCENT1, 2)
        return

    if not state_result:
        # countdown ring
        angle = int(360 * (1 - t_remaining/3))
        overlay = img.copy()
        cv2.ellipse(overlay, (cx, CANVAS_H//2), (70,70), -90, 0, angle,
                    ACCENT1, 8)
        cv2.addWeighted(overlay, 0.9, img, 0.1, 0, img)
        num = max(0, 3 - int(t_remaining))
        put_text_centered(img, str(num) if num > 0 else "GO!",
                          cx, CANVAS_H//2, cv2.FONT_HERSHEY_DUPLEX, 2.8,
                          ACCENT1 if num > 0 else WIN_COL, 4)
        put_text_centered(img, "Show your move!",
                          cx, CANVAS_H//2+80, cv2.FONT_HERSHEY_PLAIN, 1.4, GREY, 2)
    else:
        col = WIN_COL if round_result=="WIN" else (LOSE_COL if round_result=="LOSE" else TIE_COL)
        put_text_centered(img, round_result,
                          cx, CANVAS_H//2-20, cv2.FONT_HERSHEY_DUPLEX, 2.2, col, 4)
        put_text_centered(img, "Press [S] for next round",
                          cx, CANVAS_H//2+50, cv2.FONT_HERSHEY_PLAIN, 1.3, GREY, 2)


def build_frame(webcam_crop, t_remaining):
    img = np.zeros((CANVAS_H, CANVAS_W, 3), dtype=np.uint8)
    draw_gradient_bg(img)

    # header bar 
    cv2.rectangle(img, (0,0), (CANVAS_W, 70), BG_MID, -1)
    put_text_shadow(img, "ROCK  ·  PAPER  ·  SCISSORS",
                    CANVAS_W//2 - 260, 48, cv2.FONT_HERSHEY_DUPLEX, 1.1, WHITE, 2)
    # accent line
    cv2.line(img, (0,70), (CANVAS_W, 70), ACCENT1, 2)

    # decorative grid dots (background texture)
    for gx in range(60, CANVAS_W, 80):
        for gy in range(100, CANVAS_H, 80):
            cv2.circle(img, (gx, gy), 1, (50,50,70), -1)

    # side panels 
    draw_panel(img, 30, 90, 430, CANVAS_H-70,
               "A  I", aiMove, score[0])
    draw_panel(img, CANVAS_W-430, 90, CANVAS_W-30, CANVAS_H-70,
               "PLAYER", playerMove, score[1], is_player=True)

    # webcam feed 
    cam_x1, cam_y1 = 460, 180
    cam_w, cam_h   = 360, 270
    cam_x2, cam_y2 = cam_x1+cam_w, cam_y1+cam_h
    webcam_resized = cv2.resize(webcam_crop, (cam_w, cam_h))
    img[cam_y1:cam_y2, cam_x1:cam_x2] = webcam_resized
    # frame around camera
    cv2.rectangle(img, (cam_x1-2, cam_y1-2), (cam_x2+2, cam_y2+2), ACCENT2, 2)
    put_text_centered(img, "YOUR CAMERA",
                      cam_x1+cam_w//2, cam_y2+20,
                      cv2.FONT_HERSHEY_PLAIN, 1.2, GREY, 1)

    # center zone 
    draw_center_zone(img, t_remaining, stateResult, roundResult)

    # footer hint 
    cv2.rectangle(img, (0, CANVAS_H-55), (CANVAS_W, CANVAS_H), BG_MID, -1)
    hints = "[S] Start / Next Round     [Q] Quit     Fist=Rock  OpenHand=Paper  Two-Fingers=Scissors"
    put_text_shadow(img, hints, 40, CANVAS_H-18,
                    cv2.FONT_HERSHEY_PLAIN, 1.1, GREY, 1)

    return img

# Main loop

while True:
    success, raw = cap.read()
    if not success:
        break

    # scale + crop webcam to a square-ish region
    scaled = cv2.resize(raw, (0,0), None, 0.875, 0.875)
    scaled = scaled[:, 80:480]          # ~400×420

    hands, _ = detector.findHands(scaled)
    t_remaining = 0

    if startGame:
        if not stateResult:
            elapsed = time.time() - initialTime
            t_remaining = elapsed

            if elapsed > 3:
                stateResult = True
                aiMove      = random.randint(1, 3)

                if hands:
                    hand    = hands[0]
                    fingers = detector.fingersUp(hand)
                    if   fingers == [0,0,0,0,0]: playerMove = 1   # rock
                    elif fingers == [1,1,1,1,1]: playerMove = 2   # paper
                    elif fingers == [0,1,1,0,0]: playerMove = 3   # scissors
                    else:                         playerMove = None
                else:
                    playerMove = None

                # determine result
                if playerMove is None:
                    roundResult = "NO HAND"
                elif playerMove == aiMove:
                    roundResult = "TIE"
                elif (playerMove==1 and aiMove==3) or \
                     (playerMove==2 and aiMove==1) or \
                     (playerMove==3 and aiMove==2):
                    roundResult = "WIN"
                    score[1] += 1
                else:
                    roundResult = "LOSE"
                    score[0] += 1

    frame = build_frame(scaled, t_remaining)
    cv2.imshow("Rock · Paper · Scissors", frame)

    key = cv2.waitKey(1)
    if key == ord('s'):
        startGame   = True
        initialTime = time.time()
        stateResult = False
        playerMove  = None
        aiMove      = None
        roundResult = ""
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# Final result
print("\n══════════════════════════════")
print(f"  Final Score  —  AI: {score[0]}  |  Player: {score[1]}")
if   score[0] > score[1]: print("  🤖  AI wins!")
elif score[1] > score[0]: print("  🎉  Player wins!")
else:                      print("  🤝  It's a Tie!")
print("══════════════════════════════\n")