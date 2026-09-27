"""
======================================================================
  utils/prepare_dataset.py
  DeafVoice AI: Finger-Wise & Multi-Hand Sign Dataset Generator
  Generates 2,000 augmented landmark samples across 10 distinct gestures
  (100 Left Hand, 100 Right Hand per class, with variations & noise)
======================================================================
"""

import os
import sys

# Ensure project root is in sys.path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import cv2
import numpy as np
from utils.gesture_classifier import CLASS_NAMES, GESTURE_LABELS, normalize_landmarks

# Base canonical 21-landmark poses for right hand
def generate_canonical_landmarks(gesture_idx, handedness="Right"):
    lm = np.zeros((21, 3), dtype=np.float32)
    # Wrist at (0.5, 0.8, 0.0)
    lm[0] = [0.5, 0.8, 0.0]

    # Palm MCPs
    lm[5] = [0.45, 0.55, 0.0]  # Index MCP
    lm[9] = [0.50, 0.53, 0.0]  # Middle MCP
    lm[13] = [0.55, 0.55, 0.0] # Ring MCP
    lm[17] = [0.60, 0.58, 0.0] # Pinky MCP
    lm[1] = [0.42, 0.72, 0.0]  # Thumb CMC
    lm[2] = [0.38, 0.65, 0.0]  # Thumb MCP

    # Base curled finger joints (default: closed fist)
    # Index
    lm[6] = [0.45, 0.48, 0.0]
    lm[7] = [0.45, 0.52, 0.0]
    lm[8] = [0.45, 0.54, 0.0]
    # Middle
    lm[10] = [0.50, 0.46, 0.0]
    lm[11] = [0.50, 0.50, 0.0]
    lm[12] = [0.50, 0.52, 0.0]
    # Ring
    lm[14] = [0.55, 0.48, 0.0]
    lm[15] = [0.55, 0.52, 0.0]
    lm[16] = [0.55, 0.54, 0.0]
    # Pinky
    lm[18] = [0.60, 0.52, 0.0]
    lm[19] = [0.60, 0.56, 0.0]
    lm[20] = [0.60, 0.58, 0.0]
    # Thumb curled
    lm[3] = [0.42, 0.60, 0.0]
    lm[4] = [0.45, 0.62, 0.0]

    # Finger extension helpers
    def extend_index():
        lm[6] = [0.45, 0.45, 0.0]
        lm[7] = [0.45, 0.35, 0.0]
        lm[8] = [0.45, 0.22, 0.0]

    def extend_middle():
        lm[10] = [0.50, 0.42, 0.0]
        lm[11] = [0.50, 0.32, 0.0]
        lm[12] = [0.50, 0.20, 0.0]

    def extend_ring():
        lm[14] = [0.55, 0.44, 0.0]
        lm[15] = [0.55, 0.34, 0.0]
        lm[16] = [0.55, 0.24, 0.0]

    def extend_pinky():
        lm[18] = [0.60, 0.48, 0.0]
        lm[19] = [0.60, 0.38, 0.0]
        lm[20] = [0.60, 0.28, 0.0]

    def extend_thumb_up():
        lm[2] = [0.35, 0.60, 0.0]
        lm[3] = [0.32, 0.45, 0.0]
        lm[4] = [0.30, 0.30, 0.0]

    def extend_thumb_out():
        lm[2] = [0.35, 0.62, 0.0]
        lm[3] = [0.28, 0.58, 0.0]
        lm[4] = [0.22, 0.55, 0.0]

    # 10 Gestures
    if gesture_idx == 0:  # HELLO (Index Finger)
        extend_index()
    elif gesture_idx == 1:  # NO (Closed Fist)
        pass  # Already curled
    elif gesture_idx == 2:  # YES (Open Palm)
        extend_thumb_out()
        extend_index()
        extend_middle()
        extend_ring()
        extend_pinky()
    elif gesture_idx == 3:  # FINE / GOOD (Thumbs Up)
        extend_thumb_up()
    elif gesture_idx == 4:  # THANK YOU (Peace / V)
        extend_index()
        extend_middle()
        # Spread apart slightly
        lm[8, 0] -= 0.03
        lm[12, 0] += 0.03
    elif gesture_idx == 5:  # HELP (Three Fingers)
        extend_index()
        extend_middle()
        extend_ring()
    elif gesture_idx == 6:  # WAIT (Four Fingers)
        extend_index()
        extend_middle()
        extend_ring()
        extend_pinky()
    elif gesture_idx == 7:  # PERFECT (OK Sign)
        # Thumb & index touch
        lm[3] = [0.40, 0.48, 0.0]
        lm[4] = [0.44, 0.42, 0.0]
        lm[6] = [0.45, 0.48, 0.0]
        lm[7] = [0.46, 0.45, 0.0]
        lm[8] = [0.45, 0.42, 0.0]
        # Middle, ring, pinky open
        extend_middle()
        extend_ring()
        extend_pinky()
    elif gesture_idx == 8:  # DOCTOR / CALL (Phone Sign)
        extend_thumb_out()
        extend_pinky()
    elif gesture_idx == 9:  # I LOVE YOU (ILY Sign)
        extend_thumb_out()
        extend_index()
        extend_pinky()

    # Mirror for Left hand if needed
    if handedness.lower() == "left":
        lm[:, 0] = 1.0 - lm[:, 0]

    return lm


def augment_landmarks(base_lm, noise_std=0.012, scale_range=(0.85, 1.15), rot_max_deg=15):
    """Adds realistic sensor noise, scaling, and rotation to landmarks."""
    lm = base_lm.copy()
    
    # Random 2D Rotation around wrist
    angle_rad = np.radians(np.random.uniform(-rot_max_deg, rot_max_deg))
    cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
    rot_mat = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
    
    wrist = lm[0, :2]
    centered = lm[:, :2] - wrist
    rotated = centered @ rot_mat.T + wrist
    lm[:, :2] = rotated

    # Random scale
    scale = np.random.uniform(scale_range[0], scale_range[1])
    lm[:, :2] = (lm[:, :2] - wrist) * scale + wrist

    # Random Gaussian jitter
    jitter = np.random.normal(0, noise_std, lm.shape)
    jitter[0] = 0.0  # Keep wrist anchored
    lm += jitter

    return lm


def render_landmark_image(landmarks, w=240, h=240, handedness="Right", title=""):
    """Renders a beautiful visual hand skeleton diagram on dark background."""
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:] = (20, 22, 28)

    # Connections
    connections = [
        (0, 1), (1, 2), (2, 3), (3, 4),        # Thumb
        (0, 5), (5, 6), (6, 7), (7, 8),        # Index
        (0, 9), (9, 10), (10, 11), (11, 12),   # Middle
        (0, 13), (13, 14), (14, 15), (15, 16), # Ring
        (0, 17), (17, 18), (18, 19), (19, 20), # Pinky
        (5, 9), (9, 13), (13, 17)              # Palm base
    ]

    pts = (landmarks[:, :2] * np.array([w, h])).astype(int)

    # Draw lines
    for p1, p2 in connections:
        cv2.line(img, tuple(pts[p1]), tuple(pts[p2]), (0, 220, 255), 2, cv2.LINE_AA)

    # Draw joints
    for i, pt in enumerate(pts):
        color = (0, 255, 128) if i in [4, 8, 12, 16, 20] else (255, 200, 50)
        radius = 5 if i in [4, 8, 12, 16, 20] else 3
        cv2.circle(img, tuple(pt), radius, color, -1, cv2.LINE_AA)

    if title:
        cv2.putText(img, title, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(img, f"Hand: {handedness}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 255, 180), 1, cv2.LINE_AA)

    return img


def build_full_dataset(num_samples_per_class=200):
    """
    Builds a 2,000 sample dataset (1,600 train, 400 test) across all 10 finger-wise classes.
    Includes 50% Left Hand and 50% Right Hand data with rotations, scalings, and noise.
    """
    X_list = []
    y_list = []

    print("[DATASET] Generating 2,000 finger-wise multi-hand gesture samples ...")
    for class_idx in range(10):
        for i in range(num_samples_per_class):
            handedness = "Right" if i % 2 == 0 else "Left"
            base_lm = generate_canonical_landmarks(class_idx, handedness=handedness)
            aug_lm = augment_landmarks(base_lm)
            feats = normalize_landmarks(aug_lm, handedness=handedness)
            X_list.append(feats)
            y_list.append(class_idx)

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int32)

    # Shuffle
    indices = np.random.permutation(len(X))
    X = X[indices]
    y = y[indices]

    # Split into 80% train (1600) and 20% test (400)
    split_idx = int(0.80 * len(X))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    # Save to numpy format in data/
    os.makedirs('data', exist_ok=True)
    np.save('data/X_train.npy', X_train)
    np.save('data/y_train.npy', y_train)
    np.save('data/X_test.npy', X_test)
    np.save('data/y_test.npy', y_test)

    print(f"  [OK] Train dataset: {X_train.shape[0]} samples, {X_train.shape[1]} features.")
    print(f"  [OK] Test dataset:  {X_test.shape[0]} samples, {X_test.shape[1]} features.")

    # Generate reference sample images for each class in sample_images/
    os.makedirs('sample_images', exist_ok=True)
    sample_files = []
    for class_idx, name in enumerate(CLASS_NAMES):
        right_lm = generate_canonical_landmarks(class_idx, handedness="Right")
        title_text = GESTURE_LABELS[class_idx]
        img = render_landmark_image(right_lm, title=title_text, handedness="Right")
        img_path = os.path.join('sample_images', f"{name}_sample.jpg")
        cv2.imwrite(img_path, img)
        sample_files.append(img_path)

    print(f"  [OK] Generated {len(sample_files)} visual reference sample images in sample_images/.")
    return X_train, y_train, X_test, y_test


if __name__ == '__main__':
    build_full_dataset()
