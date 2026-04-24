"""
╔══════════════════════════════════════════════════════════════════════════════╗
║   HAND GESTURE VIRTUAL MOUSE  — Extended v3.0                               ║
║   New Features Added:                                                        ║
║     1. Volume Control      (Left hand pinch-distance maps to volume)         ║
║     2. Screenshots         (Left hand open palm takes a screenshot)          ║
║     3. Kalman Smoothing    (Zero-lag mathematical cursor smoothing)          ║
╚══════════════════════════════════════════════════════════════════════════════╝

INSTALL:
  pip install pycaw comtypes filterpy pystray pillow opencv-python mediapipe pyautogui numpy

LEFT HAND → Shortcuts & Control
  Index only          → Ctrl+Z  (Undo)
  Pinky only          → Ctrl+S  (Save)
  Index+Pinky         → Alt+Tab (Switch window)
  Index+Middle        → Ctrl+C  (Copy)
  Ring only           → Ctrl+V  (Paste)
  All 5 fingers UP    → TAKE SCREENSHOT
  Pinch (Thumb+Index) → SYSTEM VOLUME CONTROL (Move fingers apart to increase)

RIGHT HAND → Cursor Control
  Index only          → Move cursor
  Thumb+Index pinch   → Left click
  Middle only         → Right click
  Index+Middle up     → Scroll
  Pinch & hold        → Drag
"""

import os, math, time, datetime, threading, urllib.request
from collections import deque

import cv2
import numpy as np
import pyautogui
import mediapipe as mp

# Feature: Kalman Filter
try:
    from filterpy.kalman import KalmanFilter
    KALMAN_OK = True
except ImportError:
    KALMAN_OK = False
    print("[WARN] filterpy not found. Falling back to Moving Average smoothing.")

# Feature: System Volume (Windows)
try:
    from ctypes import cast, POINTER
    from comtypes import CLSCTX_ALL
    from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
    devices = AudioUtilities.GetSpeakers()
    interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
    sys_volume = cast(interface, POINTER(IAudioEndpointVolume))
    VOL_OK = True
except Exception as e:
    VOL_OK = False
    print(f"[WARN] Volume control disabled (pycaw error: {e})")

# Feature: System Tray
try:
    import pystray
    from PIL import Image as PILImage, ImageDraw as PILDraw
    TRAY_OK = True
except ImportError:
    TRAY_OK = False

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
class Config:
    CAM_ID     = 0
    CAM_WIDTH  = 640
    CAM_HEIGHT = 480

    MODEL_PATH = "gesture_recognizer.task"
    MODEL_URL  = "https://storage.googleapis.com/mediapipe-models/gesture_recognizer/gesture_recognizer/float16/1/gesture_recognizer.task"

    BOX_LEFT, BOX_TOP, BOX_RIGHT, BOX_BOTTOM = 100, 80, 540, 400

    # Moving Average Window (fallback)
    SMOOTH_N = 2

    PINCH_THRESHOLD  = 0.055
    CLICK_COOLDOWN   = 0.35
    DRAG_HOLD_FRAMES = 8
    SCROLL_SPEED     = 25
    SCROLL_DEADZONE  = 0.03

    SHOW_SKELETON = True
    SHOW_FPS      = True
    SHOW_BOX      = True
    SHOW_GESTURE  = True

class LM:
    THUMB_TIP  = 4
    INDEX_MCP  = 5;  INDEX_PIP  = 6;  INDEX_TIP  = 8
    MIDDLE_PIP = 10; MIDDLE_TIP = 12
    RING_PIP   = 14; RING_TIP   = 16
    PINKY_PIP  = 18; PINKY_TIP  = 20

def dist(p1, p2):
    return math.hypot(p2[0]-p1[0], p2[1]-p1[1])

def clamp_map(v, a, b, c, d):
    v = max(a, min(b, v))
    return c + (v - a) / (b - a) * (d - c)

def ensure_model(cfg: Config):
    if not os.path.exists(cfg.MODEL_PATH):
        urllib.request.urlretrieve(cfg.MODEL_URL, cfg.MODEL_PATH)

# ─────────────────────────────────────────────────────────────────────────────
# HAND DETECTOR
# ─────────────────────────────────────────────────────────────────────────────
class HandDetector:
    CONNECTIONS = [
        (0,1),(1,2),(2,3),(3,4),
        (0,5),(5,6),(6,7),(7,8),
        (0,9),(9,10),(10,11),(11,12),
        (0,13),(13,14),(14,15),(15,16),
        (0,17),(17,18),(18,19),(19,20),
        (5,9),(9,13),(13,17),
    ]

    def __init__(self, cfg: Config):
        ensure_model(cfg)
        opts = mp.tasks.vision.GestureRecognizerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=cfg.MODEL_PATH),
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            num_hands=2,
            min_hand_detection_confidence=0.70,
            min_hand_presence_confidence=0.70,
            min_tracking_confidence=0.65,
        )
        self._rec = mp.tasks.vision.GestureRecognizer.create_from_options(opts)
        self._ts  = 0

    def process(self, bgr_frame):
        h, w = bgr_frame.shape[:2]
        rgb  = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        self._ts += 1
        result = self._rec.recognize_for_video(
            mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), self._ts
        )
        hands = {}
        for lms, edness in zip(result.hand_landmarks, result.handedness):
            role = "cursor" if edness[0].category_name == "Left" else "shortcut"
            hands[role] = [(int(lm.x * w), int(lm.y * h)) for lm in lms]

        if "cursor" not in hands and "shortcut" in hands:
            hands["cursor"] = hands.pop("shortcut")
        return hands, result

    @staticmethod
    def draw_skeleton(frame, result, fw, fh):
        for hand_lms in result.hand_landmarks:
            pts = [(int(lm.x * fw), int(lm.y * fh)) for lm in hand_lms]
            for a, b in HandDetector.CONNECTIONS: cv2.line(frame, pts[a], pts[b], (0, 200, 80), 2)
            for p in pts:
                cv2.circle(frame, p, 4, (255, 255, 255), -1); cv2.circle(frame, p, 4, (0, 150, 50), 1)

    @staticmethod
    def fingers_up(lm, is_right: bool = True) -> list:
        if lm is None: return [0]*5
        thumb = ((1 if lm[LM.THUMB_TIP][0] < lm[LM.INDEX_MCP][0] else 0) if is_right
                 else (1 if lm[LM.THUMB_TIP][0] > lm[LM.INDEX_MCP][0] else 0))
        tb = [(LM.INDEX_TIP, LM.INDEX_PIP), (LM.MIDDLE_TIP, LM.MIDDLE_PIP),
              (LM.RING_TIP,  LM.RING_PIP),  (LM.PINKY_TIP,  LM.PINKY_PIP)]
        return [thumb] + [1 if lm[t][1] < lm[p][1] else 0 for t, p in tb]

# ─────────────────────────────────────────────────────────────────────────────
# KALMAN SMOOTHER
# ─────────────────────────────────────────────────────────────────────────────
class KalmanSmoother:
    def __init__(self):
        if not KALMAN_OK: return
        self.kf = KalmanFilter(dim_x=4, dim_z=2)
        # State: [x, y, dx, dy]
        self.kf.F = np.array([[1,0,1,0], [0,1,0,1], [0,0,1,0], [0,0,0,1]], dtype=float)
        self.kf.H = np.array([[1,0,0,0], [0,1,0,0]], dtype=float)
        self.kf.P *= 1000.  # high initial uncertainty
        self.kf.R *= 2.0    # measurement noise
        self.kf.Q *= 1.0    # process noise constraints
        self._init = False

    def update(self, x, y):
        if not KALMAN_OK: return int(x), int(y)
        if not self._init:
            self.kf.x = np.array([[x], [y], [0.], [0.]])
            self._init = True
        self.kf.predict()
        self.kf.update(np.array([[x], [y]]))
        return int(self.kf.x[0,0]), int(self.kf.x[1,0])

# ─────────────────────────────────────────────────────────────────────────────
# MOUSE CONTROLLER 
# ─────────────────────────────────────────────────────────────────────────────
class MouseController:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.sw, self.sh = pyautogui.size()
        self._bx, self._by = deque(maxlen=cfg.SMOOTH_N), deque(maxlen=cfg.SMOOTH_N)
        self.kalman = KalmanSmoother()
        self._lct, self._drg, self._psy = 0.0, False, None
        pyautogui.FAILSAFE, pyautogui.PAUSE = False, 0.0

    def _sm(self, x, y):
        if KALMAN_OK:
            return self.kalman.update(x, y)
        self._bx.append(x); self._by.append(y)
        return int(np.mean(self._bx)), int(np.mean(self._by))

    def _sc(self, fx, fy):
        c = self.cfg
        return (clamp_map(fx, c.BOX_LEFT, c.BOX_RIGHT,  0, self.sw),
                clamp_map(fy, c.BOX_TOP,  c.BOX_BOTTOM, 0, self.sh))

    def _ok(self): return time.time() - self._lct >= self.cfg.CLICK_COOLDOWN
    def _end_drag(self):
        if self._drg: pyautogui.mouseUp(button="left"); self._drg = False

    def move(self, fx, fy):       pyautogui.moveTo(*self._sm(*self._sc(fx, fy))); self._end_drag()
    def left_click(self, fx, fy):
        if self._ok(): pyautogui.click(*self._sm(*self._sc(fx, fy)), button="left"); self._lct = time.time(); self._end_drag()
    def right_click(self, fx, fy):
        if self._ok(): pyautogui.click(*self._sm(*self._sc(fx, fy)), button="right"); self._lct = time.time()
    def drag(self, fx, fy):
        sx, sy = self._sm(*self._sc(fx, fy))
        if not self._drg: pyautogui.mouseDown(button="left"); self._drg = True
        pyautogui.moveTo(sx, sy)
    def scroll(self, fy, fh):
        if self._psy is None: self._psy = fy; return
        d = self._psy - fy; self._psy = fy
        if abs(d) >= fh * self.cfg.SCROLL_DEADZONE: pyautogui.scroll((1 if d > 0 else -1) * self.cfg.SCROLL_SPEED)
    def release_scroll(self): self._psy = None
    def release_drag(self):   self._end_drag()

# ─────────────────────────────────────────────────────────────────────────────
# LEFT HAND EXTENSIONS (Volume, Screenshot, Shortcuts)
# ─────────────────────────────────────────────────────────────────────────────
class ShortcutManager:
    MAP = {
        (1,0,0,0): (("ctrl","z"),   "Ctrl+Z Undo"),
        (1,1,0,0): (("ctrl","c"),   "Ctrl+C Copy"),
        (0,0,1,0): (("ctrl","v"),   "Ctrl+V Paste"),
        (0,0,0,1): (("ctrl","s"),   "Ctrl+S Save"),
        (1,0,0,1): (("alt","tab"),  "Alt+Tab Switch"),
    }
    def __init__(self):
        self._last_t, self._last_key, self._label, self._label_t = 0.0, (), "", 0.0

    def trigger(self, fingers: list, lm, fw: int) -> str:
        now = time.time()
        
        # Screenshot: ALL FIVE fingers
        if fingers == [1, 1, 1, 1, 1]:
            if now - self._last_t > 2.0:
                fname = f"screenshot_{datetime.datetime.now():%H%M%S}.png"
                pyautogui.screenshot(fname)
                self._last_t = now
                self._label_t, self._label = now, f"SCREENSHOT -> {fname}"
                return self._label
            return ""

        # Volume Control: Pinch (Thumb & Index)
        pinch_dist = dist(lm[LM.THUMB_TIP], lm[LM.INDEX_TIP]) / fw
        if not any(fingers[2:]) and VOL_OK:  # Mid, Ring, Pinky down
            # Map distance (0.02 to 0.2) to Volume (-65dB to 0dB)
            vol = clamp_map(pinch_dist, 0.02, 0.18, -65.25, 0.0)
            sys_volume.SetMasterVolumeLevel(vol, None)
            
            # Show percentage string for HUD
            vol_perc = int(clamp_map(vol, -65.25, 0.0, 0, 100))
            return f"VOL: {vol_perc}%"

        # Normal Keyboard Shortcuts
        key = tuple(fingers[1:])
        if key not in self.MAP: return ""
        if key == self._last_key and now - self._last_t < 1.0: return ""
        
        hotkey, label = self.MAP[key]
        pyautogui.hotkey(*hotkey)
        self._last_t, self._last_key = now, key
        self._label, self._label_t = label, now
        return label

    @property
    def feedback(self) -> str: return self._label if time.time() - self._label_t < 1.5 else ""

# ─────────────────────────────────────────────────────────────────────────────
# DRAWING CANVAS & CLASSIFIER & TRAY & HUD
# ─────────────────────────────────────────────────────────────────────────────
class GestureClassifier:
    def __init__(self, cfg: Config, fw: int): self.cfg, self.fw, self._pf = cfg, fw, 0
    def classify(self, fingers: list, lm):
        if lm is None: self._pf = 0; return "IDLE"
        _, index, middle, ring, pinky = fingers
        pinching = dist(lm[LM.THUMB_TIP], lm[LM.INDEX_TIP]) / self.fw < self.cfg.PINCH_THRESHOLD
        self._pf  = self._pf + 1 if pinching else 0
        if pinching and self._pf >= self.cfg.DRAG_HOLD_FRAMES: return "PINCH_DRAG"
        if pinching: return "LEFT_CLICK"
        if middle==1 and sum(fingers[1:])==1: return "RIGHT_CLICK"
        if index==1 and middle==1 and ring==0 and pinky==0: return "SCROLL"
        if index==1 and sum(fingers[1:])==1: return "MOVE"
        return "IDLE"

class DrawingCanvas:
    COLORS, NAMES = [(0,255,0),(50,50,255),(0,0,255),(0,220,220),(220,220,0),(255,255,255)], ["Green","Blue","Red","Cyan","Yellow","White"]
    def __init__(self, fw, fh):
        self.canvas, self._on, self._ci, self._prev, self._ct = np.zeros((fh, fw, 3), dtype=np.uint8), False, 0, None, None
    @property
    def active(self): return self._on
    def toggle(self): self._on = not self._on; self._prev = None; return self._on
    def next_color(self): self._ci = (self._ci + 1) % len(self.COLORS); return self.NAMES[self._ci]
    def clear(self): self.canvas[:] = 0; self._prev = None
    def update(self, f, lm):
        if not self._on or lm is None: self._prev = self._ct = None; return
        fist = sum(f[1:]) == 0
        if fist:
            self._ct = self._ct or time.time()
            if time.time() - self._ct >= 1.5: self.clear(); self._ct = None
            self._prev = None; return
        self._ct = None
        if f[1]==1 and f[2]==0:
            if self._prev: cv2.line(self.canvas, self._prev, lm[LM.INDEX_TIP], self.COLORS[self._ci], 8)
            self._prev = lm[LM.INDEX_TIP]
        else: self._prev = None
    def overlay(self, frame):
        m = self.canvas.any(axis=2); r = frame.copy()
        r[m] = cv2.addWeighted(frame, 0.15, self.canvas, 0.85, 0)[m]; return r
    def save(self): fname = f"canvas_{datetime.datetime.now():%H%M%S}.png"; cv2.imwrite(fname, self.canvas); return fname

class TrayManager:
    def __init__(self, app): self._app, self._icon = app, None
    def start(self):
        if not TRAY_OK: return
        i = PILImage.new("RGBA", (64, 64), (0, 0, 0, 0)); d = PILDraw.Draw(i); c = (0, 200, 120)
        d.ellipse([10, 24, 54, 58], fill=c)
        for x, top in [(14,6),(22,2),(30,4),(38,8)]: d.rectangle([x, top, x+6, 30], fill=c)
        d.rectangle([6, 26, 14, 38], fill=c)
        m = pystray.Menu(pystray.MenuItem("Pause/Resume", lambda *a: self._app.toggle_pause()),
                         pystray.MenuItem("Toggle Canvas", lambda *a: self._app.canvas.toggle()),
                         pystray.MenuItem("Quit", lambda *a: [self._icon.stop(), self._app.stop()]))
        self._icon = pystray.Icon("GM", i, "Gesture Mouse", m)
        threading.Thread(target=self._icon.run, daemon=True).start()
    def stop(self):
        if self._icon:
            try: self._icon.stop()
            except: pass

class HUD:
    G_COLORS = {"MOVE":(0,255,0), "LEFT_CLICK":(0,200,255), "RIGHT_CLICK":(255,100,0), "SCROLL":(200,0,255), "PINCH_DRAG":(0,100,255), "IDLE":(120,120,120)}
    def __init__(self, cfg, fw, fh): self.cfg, self.fw, self.fh = cfg, fw, fh
    def draw(self, fr, gesture, curr_f, fps, canvas, s_fb, s_f, paused):
        if paused:
            cv2.rectangle(fr,(0,0),(self.fw,self.fh),(0,0,0),8)
            cv2.putText(fr,"PAUSED",(self.fw//2-70,self.fh//2),cv2.FONT_HERSHEY_DUPLEX,0.85,(0,120,255),2); return
        if canvas.active:
            cv2.rectangle(fr,(0,0),(self.fw,45),(20,20,20),-1)
            cv2.putText(fr,f"CANVAS [{canvas.NAMES[canvas._ci]}]",(10,30),cv2.FONT_HERSHEY_DUPLEX,0.75,canvas.COLORS[canvas._ci],2)
        else:
            if self.cfg.SHOW_BOX: cv2.rectangle(fr,(self.cfg.BOX_LEFT,self.cfg.BOX_TOP),(self.cfg.BOX_RIGHT,self.cfg.BOX_BOTTOM),(50,50,200),2)
            if self.cfg.SHOW_GESTURE: cv2.circle(fr,(28,28),14,self.G_COLORS.get(gesture,(200,200,200)),-1); cv2.putText(fr,gesture,(50,36),cv2.FONT_HERSHEY_DUPLEX,0.75,self.G_COLORS.get(gesture,(200,200,200)),2)
        if self.cfg.SHOW_FPS: cv2.putText(fr,f"FPS:{fps:04.1f}",(self.fw-110,28),cv2.FONT_HERSHEY_SIMPLEX,0.6,(200,255,200),2)
        if s_fb: cv2.rectangle(fr,(0,50),(self.fw,90),(20,20,20),-1); cv2.putText(fr,f" {s_fb}",(10,78),cv2.FONT_HERSHEY_DUPLEX,0.7,(0,220,255),2)
        for i,st in enumerate(curr_f):
            x, y = 10+i*36, self.fh-15; cv2.circle(fr,(x+12,y-10),10,(0,255,80) if st else (60,60,60),-1); cv2.putText(fr,"TIMRP"[i],(x+6,y-5),cv2.FONT_HERSHEY_SIMPLEX,0.4,(0,0,0),1)
        if any(s_f):
            for i,st in enumerate(s_f):
                x, y = self.fw-200+i*36, self.fh-15; cv2.circle(fr,(x+12,y-10),10,(200,100,255) if st else (60,60,60),-1); cv2.putText(fr,"TIMRP"[i],(x+6,y-5),cv2.FONT_HERSHEY_SIMPLEX,0.4,(0,0,0),1)

# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
class App:
    def __init__(self):
        self.cfg = Config()
        self.cap = cv2.VideoCapture(self.cfg.CAM_ID, cv2.CAP_DSHOW) or cv2.VideoCapture(self.cfg.CAM_ID)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.cfg.CAM_WIDTH); self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.cfg.CAM_HEIGHT); self.cap.set(cv2.CAP_PROP_FPS, 60)
        self.fw, self.fh = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.detector = HandDetector(self.cfg); self.clf = GestureClassifier(self.cfg, self.fw)
        self.mouse = MouseController(self.cfg); self.sc = ShortcutManager()
        self.canvas = DrawingCanvas(self.fw, self.fh); self.hud = HUD(self.cfg, self.fw, self.fh)
        self.tray = TrayManager(self); self.ps = False; self.runb = True; self.pt = time.time()

    def toggle_pause(self): self.ps = not self.ps; self.mouse.release_drag()
    def stop(self): self.runb = False

    def _dispatch(self, gesture, lm):
        if not lm: self.mouse.release_drag(); return
        ix, iy = lm[LM.INDEX_TIP]
        if gesture=="MOVE": self.mouse.move(ix, iy); self.mouse.release_scroll()
        elif gesture=="LEFT_CLICK": self.mouse.left_click(ix, iy); self.mouse.release_scroll()
        elif gesture=="RIGHT_CLICK": self.mouse.right_click(ix, iy); self.mouse.release_scroll()
        elif gesture=="PINCH_DRAG": self.mouse.drag(ix, iy); self.mouse.release_scroll()
        elif gesture=="SCROLL":
            avg_y = (iy + lm[LM.MIDDLE_TIP][1]) // 2
            self.mouse.scroll(avg_y, self.fh); self.mouse.release_drag()
        else: self.mouse.release_drag(); self.mouse.release_scroll()

    def run(self):
        self.tray.start()
        print("  Gesture Mouse v3.0 — LIVE")
        while self.runb:
            ret, fr = self.cap.read()
            if not ret: continue
            fr = cv2.flip(fr, 1)

            if self.ps:
                self.hud.draw(fr, "IDLE", [0]*5, 0, self.canvas, "", [0]*5, True)
                cv2.imshow("GM", fr)
                if cv2.waitKey(1) & 0xFF == ord('q'): break
                continue

            hands, res = self.detector.process(fr)
            clm, slm = hands.get("cursor"), hands.get("shortcut")
            cf, sf = HandDetector.fingers_up(clm, True), HandDetector.fingers_up(slm, False)

            if self.canvas.active: self.canvas.update(cf, clm)
            else: self._dispatch(self.clf.classify(cf, clm), clm)

            s_fb = self.sc.trigger(sf, slm, self.fw) if slm else ""
            if not s_fb: s_fb = self.sc.feedback

            if self.cfg.SHOW_SKELETON: HandDetector.draw_skeleton(fr, res, self.fw, self.fh)
            if self.canvas.active: fr = self.canvas.overlay(fr)

            fps = 1.0 / max(time.time() - self.pt, 1e-9); self.pt = time.time()
            self.hud.draw(fr, "IDLE" if self.canvas.active else self.clf.classify(cf,clm), cf, fps, self.canvas, s_fb, sf, False)

            cv2.imshow("GM", fr)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'): break
            elif key == ord('p'): self.toggle_pause()
            elif key == ord('d'): self.canvas.toggle()
            elif key == ord('c') and self.canvas.active: self.canvas.next_color()
            elif key == ord('s') and self.canvas.active: self.canvas.save()

        self.mouse.release_drag(); self.tray.stop(); self.cap.release(); cv2.destroyAllWindows()

if __name__ == "__main__": App().run()
