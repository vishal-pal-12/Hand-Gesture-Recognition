"""
======================================================================
  utils/gesture_classifier.py
  DeafVoice AI -- Robust Finger-State & Landmark Classifier
  Supports Left & Right Hands, Multi-Hand, and Zero-Overlap Logic
======================================================================
"""

import numpy as np

# 10 Non-Overlapping Finger-Based Sign Language Classes
CLASS_NAMES = [
    'index_hello',
    'fist_no',
    'palm_yes',
    'thumbs_fine',
    'peace_thanks',
    'three_help',
    'four_water',
    'ok_perfect',
    'call_doctor',
    'ily_love'
]

GESTURE_LABELS = [
    'HELLO (Index Finger)',
    'NO (Closed Fist)',
    'YES (Open Palm)',
    'FINE / GOOD (Thumbs Up)',
    'THANK YOU (Peace / V)',
    'HELP (Three Fingers)',
    'WATER (Four Fingers)',
    'PERFECT (OK Sign)',
    'DOCTOR / CALL (Phone Sign)',
    'I LOVE YOU (ILY Sign)'
]

DEAF_SPOKEN_PHRASES = [
    "Hello, Nice to meet you",
    "No, Please stop",
    "Yes, I agree and understand",
    "I am fine, everything is good",
    "Thank you very much",
    "Please help me, I need assistance",
    "I need water to drink, please",
    "Everything is perfect and all good",
    "I need a doctor or call someone",
    "I love you, Goodbye"
]

GESTURE_COLORS = [
    (0, 240, 255),    # Cyan - Hello
    (50, 50, 240),    # Red - No
    (50, 220, 50),    # Green - Yes
    (0, 200, 255),    # Yellow/Gold - Fine/Good
    (255, 140, 0),    # Orange - Thank You
    (200, 50, 220),   # Purple - Help
    (240, 180, 50),   # Blue - Water
    (50, 240, 150),   # Mint - Perfect
    (0, 100, 255),    # Coral - Doctor
    (255, 105, 180)   # Pink - I Love You
]

CLASS_TO_IDX = {name: i for i, name in enumerate(CLASS_NAMES)}
IDX_TO_CLASS = {i: name for i, name in enumerate(CLASS_NAMES)}


def normalize_landmarks(landmarks_21x3, handedness="Right"):
    """
    Normalizes 21 3D landmarks relative to wrist (lm 0) and palm scale.
    If handedness is 'Left', flips X coordinates so Left & Right hands produce
    IDENTICAL feature representations!
    Returns:
        features_63: 1D array of 63 float values.
    """
    coords = landmarks_21x3.copy()
    wrist = coords[0].copy()

    # Center at wrist
    coords -= wrist

    # Symmetric mirroring: If Left hand, flip X so it aligns with Right hand
    if handedness.lower() == "left":
        coords[:, 0] = -coords[:, 0]

    # Scale normalization by distance between wrist (0) and middle finger MCP (9)
    palm_size = np.linalg.norm(coords[9])
    if palm_size < 1e-4:
        palm_size = 1.0
    coords /= palm_size

    return coords.flatten().astype(np.float32)


def get_finger_states(landmarks_21x3, handedness="Right"):
    """
    Calculates extension states for all 5 fingers: [Thumb, Index, Middle, Ring, Pinky]
    Returns:
        states: dict with boolean flags for each finger, thumb direction, and OK distance.
    """
    lm = landmarks_21x3
    wrist = lm[0]

    # Euclidean distance helper
    def dist(i, j):
        return np.linalg.norm(lm[i] - lm[j])

    # 1. Four main fingers (Index: 8, Middle: 12, Ring: 16, Pinky: 20)
    # A finger is OPEN if distance to wrist is significantly larger than PIP to wrist
    index_open = dist(8, 0) > 1.25 * dist(6, 0) and (lm[8, 1] < lm[6, 1] + 0.05)
    middle_open = dist(12, 0) > 1.25 * dist(10, 0) and (lm[12, 1] < lm[10, 1] + 0.05)
    ring_open = dist(16, 0) > 1.25 * dist(14, 0) and (lm[16, 1] < lm[14, 1] + 0.05)
    pinky_open = dist(20, 0) > 1.25 * dist(18, 0) and (lm[20, 1] < lm[18, 1] + 0.05)

    # 2. Thumb Analysis
    # Thumbs Up: Thumb tip is higher than MCP and IP, pointing upwards, while others curled
    thumb_up = (lm[4, 1] < lm[3, 1]) and (lm[3, 1] < lm[2, 1]) and (dist(4, 0) > 1.15 * dist(2, 0))
    
    # Thumb Extended Outwards:
    if handedness.lower() == "left":
        thumb_out = (lm[4, 0] > lm[2, 0] + 0.04) or (dist(4, 0) > 1.20 * dist(2, 0))
    else:
        thumb_out = (lm[4, 0] < lm[2, 0] - 0.04) or (dist(4, 0) > 1.20 * dist(2, 0))

    # OK Circle: Thumb tip (4) and Index tip (8) are touching
    ok_dist = dist(4, 8)
    is_ok_circle = ok_dist < 0.07

    # Count of main extended fingers
    main_count = sum([index_open, middle_open, ring_open, pinky_open])

    return {
        'thumb_up': bool(thumb_up),
        'thumb_out': bool(thumb_out),
        'index': bool(index_open),
        'middle': bool(middle_open),
        'ring': bool(ring_open),
        'pinky': bool(pinky_open),
        'ok_circle': bool(is_ok_circle),
        'main_count': int(main_count)
    }


def classify_finger_gesture(landmarks_21x3, handedness="Right", model=None):
    """
    Classifies the hand gesture using geometric finger states and/or neural model.
    Works identically for Left and Right hands.
    Returns:
        pred_idx: int (0 to 9)
        pred_label: str
        spoken_phrase: str
        confidence: float (0.0 to 1.0)
        probabilities: np.ndarray (10,)
    """
    f = get_finger_states(landmarks_21x3, handedness=handedness)
    probs = np.zeros(10, dtype=np.float32)

    # Deterministic geometric decision tree (eliminates overlapping):
    
    # 1. Thumbs Up -> FINE / GOOD (Sign 3)
    if f['thumb_up'] and f['main_count'] == 0:
        pred_idx = 3
        confidence = 0.98

    # 2. OK Sign -> PERFECT (Sign 7)
    elif f['ok_circle'] and f['middle'] and f['ring'] and f['pinky']:
        pred_idx = 7
        confidence = 0.97

    # 3. Phone / Call Sign -> DOCTOR (Sign 8): Thumb & Pinky extended
    elif (f['thumb_out'] or f['thumb_up']) and f['pinky'] and not f['index'] and not f['middle'] and not f['ring']:
        pred_idx = 8
        confidence = 0.96

    # 4. I Love You -> I LOVE YOU (Sign 9): Thumb, Index, Pinky extended
    elif (f['thumb_out'] or f['thumb_up']) and f['index'] and f['pinky'] and not f['middle'] and not f['ring']:
        pred_idx = 9
        confidence = 0.97

    # 5. Closed Fist -> NO (Sign 1): All fingers curled
    elif f['main_count'] == 0 and not f['thumb_up']:
        pred_idx = 1
        confidence = 0.95

    # 6. Index Finger Pointing -> HELLO (Sign 0): Only Index open
    elif f['index'] and not f['middle'] and not f['ring'] and not f['pinky']:
        pred_idx = 0
        confidence = 0.96

    # 7. Peace / V Sign -> THANK YOU (Sign 4): Index + Middle open
    elif f['index'] and f['middle'] and not f['ring'] and not f['pinky']:
        pred_idx = 4
        confidence = 0.96

    # 8. Three Fingers -> HELP (Sign 5): Index + Middle + Ring open
    elif f['index'] and f['middle'] and f['ring'] and not f['pinky']:
        pred_idx = 5
        confidence = 0.95

    # 9. Four Fingers -> WATER (Sign 6): 4 fingers open, thumb curled
    elif f['main_count'] == 4 and not f['thumb_out'] and not f['thumb_up']:
        pred_idx = 6
        confidence = 0.96

    # 10. Open Palm -> YES (Sign 2): All 5 open
    elif f['main_count'] >= 4 and (f['thumb_out'] or f['thumb_up']):
        pred_idx = 2
        confidence = 0.98

    else:
        # Fallback based on finger count
        if f['main_count'] == 1 and f['index']:
            pred_idx = 0
        elif f['main_count'] == 2 and f['index'] and f['middle']:
            pred_idx = 4
        elif f['main_count'] == 3:
            pred_idx = 5
        elif f['main_count'] >= 4:
            pred_idx = 2
        else:
            pred_idx = 1
        confidence = 0.85

    # If neural model is provided, compute model predictions and blend
    if model is not None:
        try:
            feats = normalize_landmarks(landmarks_21x3, handedness=handedness).reshape(1, -1)
            model_probs = model.predict(feats, verbose=0)[0]
            # Blend geometric and neural probabilities
            geo_probs = np.zeros(10, dtype=np.float32)
            geo_probs[pred_idx] = 1.0
            probs = 0.60 * geo_probs + 0.40 * model_probs
            probs /= np.sum(probs)
            pred_idx = int(np.argmax(probs))
            confidence = float(probs[pred_idx])
        except Exception:
            probs[pred_idx] = confidence
            for i in range(10):
                if i != pred_idx:
                    probs[i] = (1.0 - confidence) / 9.0
    else:
        probs[pred_idx] = confidence
        for i in range(10):
            if i != pred_idx:
                probs[i] = (1.0 - confidence) / 9.0

    return pred_idx, GESTURE_LABELS[pred_idx], DEAF_SPOKEN_PHRASES[pred_idx], confidence, probs
