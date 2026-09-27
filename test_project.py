# test_project.py -- DeafVoice AI Unit Test Suite
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

from models.architecture import build_paper_cnn
from utils.helpers import (
    CLASS_NAMES,
    GESTURE_LABELS,
    DEAF_SPOKEN_PHRASES,
    preprocess_image
)
from utils.hand_detector import HandDetector
from utils.assistive_comm import (
    SentenceBuilder,
    GestureStabilityTracker,
    VOCABULARY_MODES,
    DEAF_DAILY_PHRASES
)


class TestDeafVoiceProject(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.project_root = os.path.dirname(os.path.abspath(__file__))
        cls.train_dir = os.path.join(cls.project_root, 'data', 'processed', 'train')
        cls.test_dir = os.path.join(cls.project_root, 'data', 'processed', 'test')

    def test_01_dataset_structure(self):
        self.assertTrue(os.path.exists(self.train_dir), "Train directory missing")
        self.assertTrue(os.path.exists(self.test_dir), "Test directory missing")
        for c in CLASS_NAMES:
            c_train = os.path.join(self.train_dir, c)
            c_test = os.path.join(self.test_dir, c)
            self.assertTrue(os.path.exists(c_train), f"Missing class in train: {c}")
            self.assertTrue(os.path.exists(c_test), f"Missing class in test: {c}")
            self.assertEqual(len(os.listdir(c_train)), 160, f"Expected 160 train images in {c}")
            self.assertEqual(len(os.listdir(c_test)), 40, f"Expected 40 test images in {c}")

    def test_02_model_architecture(self):
        model = build_paper_cnn(input_shape=(100, 100, 3), num_classes=10)
        self.assertIn(model.count_params(), [166042, 166266], "Parameter count mismatch")
        dummy = tf.zeros((2, 100, 100, 3))
        out = model(dummy)
        self.assertEqual(out.shape, (2, 10))

    def test_03_preprocessing(self):
        dummy = np.ones((120, 160, 3), dtype=np.uint8) * 200
        tensor = preprocess_image(dummy, img_size=100)
        self.assertEqual(tensor.shape, (1, 100, 100, 3))
        self.assertAlmostEqual(float(tensor.max()), 200.0 / 255.0, places=4)

    def test_04_hand_detector_roi(self):
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        roi = HandDetector.get_fixed_roi(dummy_frame, size=280)
        self.assertEqual(len(roi), 4)
        x, y, w, h = roi
        self.assertTrue(0 <= x < 640)
        self.assertTrue(0 <= y < 480)

    def test_05_sample_images_exist(self):
        sample_dir = os.path.join(self.project_root, 'sample_images')
        self.assertTrue(os.path.exists(sample_dir))
        for c in CLASS_NAMES:
            sample_file = os.path.join(sample_dir, f"{c}_sample.jpg")
            self.assertTrue(os.path.exists(sample_file), f"Missing sample file: {sample_file}")

    def test_06_deaf_assistive_engine(self):
        self.assertEqual(len(GESTURE_LABELS), 10)
        self.assertEqual(len(DEAF_SPOKEN_PHRASES), 10)
        sb = SentenceBuilder()
        token = sb.add_gesture(0)  # HELLO
        self.assertTrue("Hello" in token)
        self.assertTrue("Hello" in sb.get_sentence())
        sb.add_gesture(3)  # THANK YOU
        self.assertTrue("Thank You" in sb.get_sentence())
        sb.backspace()
        self.assertFalse("Thank You" in sb.get_sentence())
        sb.clear()
        self.assertEqual(sb.get_sentence(), "")

    def test_07_stability_tracker(self):
        tracker = GestureStabilityTracker(required_frames=5, cooldown_seconds=0.5)
        # Test progress increment
        for _ in range(4):
            committed, progress = tracker.update(detected_idx=0, confidence=0.85)
            self.assertIsNone(committed)
        # 5th frame commits
        committed, progress = tracker.update(detected_idx=0, confidence=0.85)
        self.assertEqual(committed, 0)
        self.assertEqual(progress, 1.0)


if __name__ == '__main__':
    unittest.main()
