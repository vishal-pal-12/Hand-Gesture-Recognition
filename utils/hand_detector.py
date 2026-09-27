# Real-time Hand Detector using MediaPipe and Fixed ROI fallback
import cv2
import numpy as np

class HandDetector:
    def __init__(self, static_mode=False, max_hands=1, min_detection_conf=0.5, min_tracking_conf=0.5):
        self.static_mode = static_mode
        self.max_hands = max_hands
        self.min_detection_conf = min_detection_conf
        self.min_tracking_conf = min_tracking_conf
        
        self.has_mediapipe = False
        try:
            import mediapipe as mp
            self.mp_hands = mp.solutions.hands
            self.mp_drawing = mp.solutions.drawing_utils
            self.mp_drawing_styles = mp.solutions.drawing_styles
            self.hands = self.mp_hands.Hands(
                static_image_mode=self.static_mode,
                max_num_hands=self.max_hands,
                min_detection_confidence=self.min_detection_conf,
                min_tracking_confidence=self.min_tracking_conf
            )
            self.has_mediapipe = True
        except Exception as e:
            print(f"[WARNING] MediaPipe Hands not available ({e}). Using fixed ROI mode.")
            self.hands = None

    def find_hands(self, frame_bgr, draw=True):
        boxes = []
        landmarks_list = []
        h, w, _ = frame_bgr.shape

        if not self.has_mediapipe or self.hands is None:
            return frame_bgr, boxes, landmarks_list

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        results = self.hands.process(frame_rgb)

        if results.multi_hand_landmarks:
            for hand_lms in results.multi_hand_landmarks:
                landmarks_list.append(hand_lms)
                if draw:
                    self.mp_drawing.draw_landmarks(
                        frame_bgr,
                        hand_lms,
                        self.mp_hands.HAND_CONNECTIONS,
                        self.mp_drawing_styles.get_default_hand_landmarks_style(),
                        self.mp_drawing_styles.get_default_hand_connections_style()
                    )

                x_coords = [lm.x for lm in hand_lms.landmark]
                y_coords = [lm.y for lm in hand_lms.landmark]

                x_min, x_max = int(min(x_coords) * w), int(max(x_coords) * w)
                y_min, y_max = int(min(y_coords) * h), int(max(y_coords) * h)

                box_w = x_max - x_min
                box_h = y_max - y_min

                pad_x = int(box_w * 0.25)
                pad_y = int(box_h * 0.25)

                x1 = max(0, x_min - pad_x)
                y1 = max(0, y_min - pad_y)
                x2 = min(w, x_max + pad_x)
                y2 = min(h, y_max + pad_y)

                boxes.append((x1, y1, x2 - x1, y2 - y1))

        return frame_bgr, boxes, landmarks_list

    @staticmethod
    def get_fixed_roi(frame_bgr, size=280):
        h, w, _ = frame_bgr.shape
        cx, cy = int(w * 0.72), int(h * 0.5)
        x = max(10, cx - size // 2)
        y = max(10, cy - size // 2)
        w_box = min(size, w - x - 10)
        h_box = min(size, h - y - 10)
        return (x, y, w_box, h_box)
