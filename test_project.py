# test_project.py -- DeafVoice AI Unit Test Suite for Finger-Wise Multi-Hand AI
import os
import sys

# Auto-redirect to project venv python if not already running in it
project_root = os.path.dirname(os.path.abspath(__file__))
venv_py = os.path.join(project_root, 'venv', 'Scripts', 'python.exe')
if os.path.exists(venv_py) and os.path.normcase(sys.executable) != os.path.normcase(venv_py):
    import subprocess
    res = subprocess.run([venv_py] + sys.argv, cwd=project_root)
    sys.exit(res.returncode)

import unittest
import numpy as np
import tensorflow as tf

from models.architecture import build_landmark_classifier
from utils.gesture_classifier import (
    CLASS_NAMES,
    GESTURE_LABELS,
    DEAF_SPOKEN_PHRASES,
    normalize_landmarks,
    classify_finger_gesture
)
from utils.prepare_dataset import generate_canonical_landmarks
from utils.hand_detector import HandDetector
from utils.assistive_comm import SentenceBuilder, GestureStabilityTracker


class TestDeafVoiceMultiHandProject(unittest.TestCase):

    def test_01_classes_and_phrases_count(self):
        self.assertEqual(len(CLASS_NAMES), 9)
        self.assertEqual(len(GESTURE_LABELS), 9)
        self.assertEqual(len(DEAF_SPOKEN_PHRASES), 9)

    def test_02_multi_hand_detector_configuration(self):
        detector = HandDetector(max_hands=2)
        self.assertEqual(detector.max_hands, 2)
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        _, hand_data = detector.find_hands(dummy_frame, draw=False)
        self.assertIsInstance(hand_data, list)

    def test_03_landmark_normalization_symmetry(self):
        # Generate canonical right hand
        right_lm = generate_canonical_landmarks(gesture_idx=0, handedness="Right")
        feats_r = normalize_landmarks(right_lm, handedness="Right")
        self.assertEqual(feats_r.shape, (63,))

        # Generate canonical left hand (mirrored)
        left_lm = generate_canonical_landmarks(gesture_idx=0, handedness="Left")
        feats_l = normalize_landmarks(left_lm, handedness="Left")
        self.assertEqual(feats_l.shape, (63,))

        # They should be symmetric and closely match
        diff = np.max(np.abs(feats_r - feats_l))
        self.assertLess(diff, 0.05, f"Left and Right hands must have symmetric feature representations! Diff: {diff}")

    def test_04_zero_overlapping_classification_all_9(self):
        for idx in range(9):
            # Test Right Hand
            lm_r = generate_canonical_landmarks(gesture_idx=idx, handedness="Right")
            pred_r, _, _, conf_r, _ = classify_finger_gesture(lm_r, handedness="Right")
            self.assertEqual(pred_r, idx, f"Right hand gesture {idx} ({GESTURE_LABELS[idx]}) misclassified as {pred_r}!")
            self.assertGreater(conf_r, 0.90)

            # Test Left Hand
            lm_l = generate_canonical_landmarks(gesture_idx=idx, handedness="Left")
            pred_l, _, _, conf_l, _ = classify_finger_gesture(lm_l, handedness="Left")
            self.assertEqual(pred_l, idx, f"Left hand gesture {idx} ({GESTURE_LABELS[idx]}) misclassified as {pred_l}!")
            self.assertGreater(conf_l, 0.90)

    def test_04b_doctor_gesture_as_hello_and_wait(self):
        for h in ["Right", "Left"]:
            fist = generate_canonical_landmarks(gesture_idx=1, handedness=h)
            
            # Solo Middle Finger -> WAIT (idx 6)
            mid = fist.copy()
            mid[10] = [0.50, 0.42, 0.0]
            mid[11] = [0.50, 0.30, 0.0]
            mid[12] = [0.50, 0.20, 0.0]
            pred_m, label_m, phrase_m, conf_m, _ = classify_finger_gesture(mid, handedness=h)
            self.assertEqual(pred_m, 6, f"{h} solo middle must be WAIT (6), got {pred_m}")
            self.assertEqual(phrase_m, "Please wait a moment")

            # Doctor Phone Sign (Thumb + Pinky) -> HELLO (idx 0)
            phone = fist.copy()
            # thumb out
            t_x = 0.22 if h == "Right" else 0.78
            phone[2] = [0.35 if h == "Right" else 0.65, 0.62, 0.0]
            phone[3] = [0.28 if h == "Right" else 0.72, 0.58, 0.0]
            phone[4] = [t_x, 0.55, 0.0]
            # pinky extended
            p_x = 0.60 if h == "Right" else 0.40
            phone[18] = [p_x, 0.48, 0.0]
            phone[19] = [p_x, 0.38, 0.0]
            phone[20] = [p_x, 0.28, 0.0]
            pred_p, label_p, phrase_p, conf_p, _ = classify_finger_gesture(phone, handedness=h)
            self.assertEqual(pred_p, 0, f"{h} phone sign must be HELLO (0), got {pred_p}")
            self.assertEqual(phrase_p, "Hello, Nice to meet you")

            # Solo Index Finger -> NOT HELLO (Removed from Hello)
            idx_x = 0.45 if h == "Right" else 0.55
            idx_lm = fist.copy()
            idx_lm[6] = [idx_x, 0.45, 0.0]
            idx_lm[7] = [idx_x, 0.35, 0.0]
            idx_lm[8] = [idx_x, 0.22, 0.0]
            pred_i, label_i, phrase_i, conf_i, _ = classify_finger_gesture(idx_lm, handedness=h)
            self.assertNotEqual(pred_i, 0, f"{h} solo index must NOT be HELLO (0)")

    def test_05_landmark_model_architecture(self):
        model = build_landmark_classifier(input_dim=63, num_classes=9)
        dummy_input = tf.zeros((2, 63))
        out = model(dummy_input)
        self.assertEqual(out.shape, (2, 9))

    def test_06_assistive_communication_sentence_builder(self):
        sb = SentenceBuilder()
        token1 = sb.add_gesture(0)  # HELLO
        token2 = sb.add_gesture(3)  # FINE
        sentence = sb.get_sentence()
        self.assertTrue("Hello" in sentence)
        sb.backspace()
        self.assertFalse("Fine" in sb.get_sentence())
        sb.clear()
        self.assertEqual(sb.get_sentence(), "")

    def test_07_stability_tracker_hold_gauge(self):
        tracker = GestureStabilityTracker(required_frames=6, cooldown_seconds=0.5)
        for _ in range(5):
            committed, progress = tracker.update(detected_idx=0, confidence=0.85)
            self.assertIsNone(committed)
        committed, progress = tracker.update(detected_idx=0, confidence=0.85)
        self.assertEqual(committed, 0)
        self.assertEqual(progress, 1.0)


if __name__ == '__main__':
    unittest.main()
