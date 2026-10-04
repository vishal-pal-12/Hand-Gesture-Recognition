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
    'four_wait',
    'ok_perfect',
    'call_doctor',
    'ily_love'
]

GESTURE_LABELS = [
    'HELLO (Index Finger)',
    'NO (Thumb Down / Fist)',
    'YES (Open Palm)',
    'FINE / GOOD (Thumbs Up)',
    'THANK YOU (Peace / V)',
    'HELP (Pinky Finger)',
    'WAIT (Middle Finger)',
    'PERFECT (OK Sign)',
    'DOCTOR / CALL (Phone / L-Shape)',
    'I LOVE YOU (ILY Sign)'
]

DEAF_SPOKEN_PHRASES = [
    "Hello, Nice to meet you",
    "No, Please stop",
    "Yes, I agree and understand",
    "I am fine, everything is good",
    "Thank you very much",
    "Please help me, I need assistance",
    "Please wait a moment",
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
    (50, 170, 240),   # Amber/Orange - Wait
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
    Uses distance-ratio geometry that is rotation-invariant and robust on webcams.
    """
    lm = landmarks_21x3
    wrist = lm[0]

    def dist(i, j):
        return float(np.linalg.norm(lm[i] - lm[j]))

    # Palm reference scale: wrist (0) to middle MCP (9)
    palm_scale = max(dist(0, 9), 0.05)

    # 1. Four main fingers: Index (8), Middle (12), Ring (16), Pinky (20)
    # A finger is open if tip is significantly farther from wrist than PIP (joint)
    # AND tip is farther from its MCP joint than PIP is from MCP
    index_open = (dist(8, 0) > 1.10 * dist(6, 0)) and (dist(8, 5) > 1.15 * dist(6, 5))
    middle_open = (dist(12, 0) > 1.10 * dist(10, 0)) and (dist(12, 9) > 1.15 * dist(10, 9))
    ring_open = (dist(16, 0) > 1.10 * dist(14, 0)) and (dist(16, 13) > 1.15 * dist(14, 13))
    pinky_open = (dist(20, 0) > 1.10 * dist(18, 0)) and (dist(20, 17) > 1.15 * dist(18, 17))

    # 2. Thumb Analysis
    # Thumbs Up: Thumb tip (4) is vertically above IP (3) and MCP (2) (smaller Y)
    thumb_up = (lm[4, 1] < lm[3, 1] - 0.015) and (lm[3, 1] < lm[2, 1] + 0.02) and (dist(4, 0) > 1.05 * dist(2, 0))

    # Thumbs Down: Thumb tip (4) is vertically below IP (3) and MCP (2) (larger Y)
    thumb_down = (lm[4, 1] > lm[3, 1] + 0.015) and (lm[3, 1] > lm[2, 1] - 0.02) and (dist(4, 0) > 1.05 * dist(2, 0))

    # Thumb extended outward horizontally away from palm
    thumb_extended = dist(4, 2) > 1.12 * dist(3, 2) and dist(4, 0) > 1.10 * dist(2, 0)
    if handedness.lower() == "left":
        thumb_out = (lm[4, 0] > lm[2, 0] + 0.02) or thumb_extended
    else:
        thumb_out = (lm[4, 0] < lm[2, 0] - 0.02) or thumb_extended

    # Pinch / OK Circle: distance between Thumb tip (4) and Index tip (8)
    ok_dist = dist(4, 8) / palm_scale
    is_ok_circle = ok_dist < 0.38

    # Count of main extended fingers (Index, Middle, Ring, Pinky)
    main_count = sum([index_open, middle_open, ring_open, pinky_open])

    return {
        'thumb_up': bool(thumb_up),
        'thumb_down': bool(thumb_down),
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
    Supports single-finger shortcuts for NO (Thumb Down), HELP (Solo Pinky), 
    WAIT (Pinch/4-fing), and DOCTOR (L-shape/Phone).
    """
    f = get_finger_states(landmarks_21x3, handedness=handedness)
    probs = np.zeros(10, dtype=np.float32)

    # 1. Thumbs Up -> FINE / GOOD (Sign 3): Solo Thumb UP (1 finger)
    if f['thumb_up'] and not f['index'] and not f['middle'] and not f['ring'] and not f['pinky']:
        pred_idx = 3
        confidence = 0.98

    # 2. Thumbs Down -> NO (Sign 1): Solo Thumb DOWN (1 finger)
    elif f['thumb_down'] and not f['index'] and not f['middle'] and not f['ring'] and not f['pinky']:
        pred_idx = 1
        confidence = 0.98

    # 3. Closed Fist -> NO (Sign 1): All fingers curled
    elif f['main_count'] == 0 and not f['thumb_up'] and not f['thumb_down']:
        pred_idx = 1
        confidence = 0.96

    # 4. Solo Pinky Finger -> HELP (Sign 5): Little Finger Only extended UP (1 finger)
    elif f['pinky'] and not f['index'] and not f['middle'] and not f['ring'] and not f['thumb_up'] and not f['thumb_down']:
        pred_idx = 5
        confidence = 0.99

    # 5. Solo Middle Finger -> WAIT (Sign 6): Middle Finger Only extended UP (1 finger)
    elif f['middle'] and not f['index'] and not f['ring'] and not f['pinky'] and not f['thumb_up'] and not f['thumb_down']:
        pred_idx = 6
        confidence = 0.99

    # 6. Solo Index Finger -> HELLO (Sign 0): Index Finger Only extended UP (1 finger)
    elif f['index'] and not f['middle'] and not f['ring'] and not f['pinky'] and not f['thumb_up'] and not f['thumb_down']:
        if f['thumb_out']:
            pred_idx = 8  # L-Shape (Thumb + Index) -> DOCTOR / CALL
            confidence = 0.97
        else:
            pred_idx = 0  # Solo Index -> HELLO
            confidence = 0.99

    # 7. Phone Sign (Thumb + Pinky) -> DOCTOR / CALL (Sign 8): Thumb + Pinky
    elif (f['thumb_out'] or f['thumb_up']) and f['pinky'] and not f['index'] and not f['middle'] and not f['ring']:
        pred_idx = 8
        confidence = 0.97

    # 8. Peace / V Sign -> THANK YOU (Sign 4): Index + Middle open (2 fingers)
    elif f['index'] and f['middle'] and not f['ring'] and not f['pinky']:
        pred_idx = 4
        confidence = 0.98

    # 9. I Love You -> I LOVE YOU (Sign 9): Thumb + Index + Pinky open
    elif (f['thumb_out'] or f['thumb_up']) and f['index'] and f['pinky'] and not f['middle'] and not f['ring']:
        pred_idx = 9
        confidence = 0.97

    # 10. OK Sign -> PERFECT (Sign 7): Thumb & Index circle, other 3 open
    elif f['ok_circle'] and f['middle'] and (f['ring'] or f['pinky']):
        pred_idx = 7
        confidence = 0.97

    # 11. Pinch Sign -> WAIT (Sign 6): Thumb & Index tips close together, others curled (1-min / 🤏)
    elif f['ok_circle'] and f['index'] and not f['middle'] and not f['ring'] and not f['pinky']:
        pred_idx = 6
        confidence = 0.97

    # 12. Four Fingers -> WAIT (Sign 6): 4 fingers open, thumb curled
    elif f['main_count'] == 4 and not f['thumb_out'] and not f['thumb_up']:
        pred_idx = 6
        confidence = 0.97

    # 13. Three Fingers -> HELP (Sign 5): Index + Middle + Ring open
    elif f['index'] and f['middle'] and f['ring'] and not f['pinky']:
        pred_idx = 5
        confidence = 0.96

    # 14. Open Palm -> YES (Sign 2): All 5 open
    elif f['main_count'] >= 4 and (f['thumb_out'] or f['thumb_up']):
        pred_idx = 2
        confidence = 0.98

    else:
        # Fallback based on finger combinations
        if f['thumb_down']:
            pred_idx = 1
        elif f['thumb_up']:
            pred_idx = 3
        elif f['main_count'] == 1 and f['middle']:
            pred_idx = 6
        elif f['main_count'] == 1 and f['index']:
            pred_idx = 0
        elif f['main_count'] == 1 and f['pinky']:
            pred_idx = 5
        elif f['main_count'] == 2 and f['index'] and f['middle']:
            pred_idx = 4
        elif f['main_count'] == 3:
            pred_idx = 5
        elif f['main_count'] >= 4:
            pred_idx = 2
        else:
            pred_idx = 1
        confidence = 0.88

    # If neural model is provided, compute model predictions and blend
    if model is not None:
        try:
            feats = normalize_landmarks(landmarks_21x3, handedness=handedness).reshape(1, -1)
            model_probs = model.predict(feats, verbose=0)[0]
            geo_probs = np.zeros(10, dtype=np.float32)
            geo_probs[pred_idx] = 1.0
            # Higher weight on deterministic geometry (0.80) to eliminate ambiguity
            probs = 0.80 * geo_probs + 0.20 * model_probs
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
