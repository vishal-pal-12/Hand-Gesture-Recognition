# utils/hand_detector.py -- Real-time Multi-Hand Detector using MediaPipe
import cv2
import numpy as np


class HandDetector:
    """Detects multiple hands (Left and Right) with 21 3D landmarks using MediaPipe."""
    def __init__(self, static_mode=False, max_hands=2, min_detection_conf=0.6, min_tracking_conf=0.6, detection_con=None, track_con=None, **kwargs):
        if detection_con is not None:
            min_detection_conf = detection_con
        if track_con is not None:
            min_tracking_conf = track_con
        self.static_mode = static_mode
        self.max_hands = max_hands
        self.min_detection_conf = min_detection_conf
        self.min_tracking_conf = min_tracking_conf
        
        self.has_mediapipe = False
        self.hands = None
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
        """
        Detects up to max_hands in frame.
        Returns:
            frame_annotated: Frame with landmarks drawn (if draw=True)
            hand_data: list of dicts for each detected hand:
                {
                    'bbox': (x, y, w, h),
                    'landmarks': np.ndarray of shape (21, 3) normalized (0-1),
                    'handedness': 'Left' or 'Right',
                    'score': float confidence
                }
        """
        hand_data = []
        h, w, _ = frame_bgr.shape

        if not self.has_mediapipe or self.hands is None:
            return frame_bgr, hand_data

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        results = self.hands.process(frame_rgb)

        if results.multi_hand_landmarks:
            for idx, hand_lms in enumerate(results.multi_hand_landmarks):
                # Handedness label
                handedness_label = "Right"
                score = 0.95
                if results.multi_handedness and idx < len(results.multi_handedness):
                    h_info = results.multi_handedness[idx].classification[0]
                    # Note: MediaPipe mirrors camera handedness, so we align with user's perspective
                    handedness_label = h_info.label
                    score = float(h_info.score)

                # Landmark array (21, 3)
                lm_array = np.array([[lm.x, lm.y, lm.z] for lm in hand_lms.landmark], dtype=np.float32)

                # Bounding box with margin
                x_coords = lm_array[:, 0] * w
                y_coords = lm_array[:, 1] * h
                pad = 20
                x_min = max(0, int(np.min(x_coords)) - pad)
                y_min = max(0, int(np.min(y_coords)) - pad)
                x_max = min(w, int(np.max(x_coords)) + pad)
                y_max = min(h, int(np.max(y_coords)) + pad)
                box_w = x_max - x_min
                box_h = y_max - y_min

                hand_data.append({
                    'bbox': (x_min, y_min, box_w, box_h),
                    'landmarks': lm_array,
                    'landmarks_norm': lm_array,
                    'handedness': handedness_label,
                    'score': score
                })

                if draw:
                    # Draw landmarks on frame
                    self.mp_drawing.draw_landmarks(
                        frame_bgr,
                        hand_lms,
                        self.mp_hands.HAND_CONNECTIONS,
                        self.mp_drawing_styles.get_default_hand_landmarks_style(),
                        self.mp_drawing_styles.get_default_hand_connections_style()
                    )

        return frame_bgr, hand_data

    @staticmethod
    def get_fixed_roi(frame, size=240):
        """Fallback fixed ROI box for when camera/mediapipe has no detection."""
        h, w, _ = frame.shape
        cx, cy = w // 2, h // 2
        x = max(0, cx - size // 2)
        y = max(0, cy - size // 2)
        return x, y, size, size
