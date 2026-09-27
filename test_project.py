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
        self.assertEqual(len(CLASS_NAMES), 10)
        self.assertEqual(len(GESTURE_LABELS), 10)
        self.assertEqual(len(DEAF_SPOKEN_PHRASES), 10)

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

    def test_04_zero_overlapping_classification_all_10(self):
        for idx in range(10):
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

    def test_05_landmark_model_architecture(self):
        model = build_landmark_classifier(input_dim=63, num_classes=10)
        dummy_input = tf.zeros((2, 63))
        out = model(dummy_input)
        self.assertEqual(out.shape, (2, 10))

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
