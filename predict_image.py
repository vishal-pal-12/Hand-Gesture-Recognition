# predict_image.py -- DeafVoice AI: Static Sign Language Image Translation
import os
import sys

# Auto-redirect to project venv python if not already running in it
project_root = os.path.dirname(os.path.abspath(__file__))
venv_py = os.path.join(project_root, 'venv', 'Scripts', 'python.exe')
if os.path.exists(venv_py) and os.path.normcase(sys.executable) != os.path.normcase(venv_py):
    import subprocess
    res = subprocess.run([venv_py] + sys.argv, cwd=project_root)
    sys.exit(res.returncode)

import cv2
import argparse
import numpy as np

from utils.helpers import (
    CLASS_NAMES,
    GESTURE_LABELS,
    DEAF_SPOKEN_PHRASES,
    GESTURE_COLORS,
    preprocess_image,
    safe_load_model
)
from utils.hand_detector import HandDetector
from utils.assistive_comm import AsyncVoiceSynthesizer

MODEL_PATH = 'models/best_model.keras'
FALLBACK_MODEL_PATH = 'models/hand_posture_cnn_final.keras'


def get_model():
    if os.path.exists(MODEL_PATH):
        return safe_load_model(MODEL_PATH)
    elif os.path.exists(FALLBACK_MODEL_PATH):
        return safe_load_model(FALLBACK_MODEL_PATH)
    else:
        raise FileNotFoundError(
            "No trained model found. Please run 'python run.py train' first."
        )


def draw_prediction_card(image_bgr, pred_idx, confidence, probabilities):
    h, w, _ = image_bgr.shape
    card_h = min(130, int(h * 0.38))
    overlay = image_bgr.copy()

    cv2.rectangle(overlay, (0, 0), (w, card_h), (18, 18, 22), -1)
    alpha = 0.85
    cv2.addWeighted(overlay, alpha, image_bgr, 1 - alpha, 0, image_bgr)

    sign_text = f"DEAF SIGN: {GESTURE_LABELS[pred_idx]}"
    spoken_text = f'SPOKEN VOICE: "{DEAF_SPOKEN_PHRASES[pred_idx]}"'
    conf_text = f"Confidence: {confidence * 100:.1f}%"
    color = GESTURE_COLORS[pred_idx]

    cv2.putText(image_bgr, sign_text, (12, 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.60, (0, 240, 255), 2, cv2.LINE_AA)
    cv2.putText(image_bgr, spoken_text, (12, 52),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, (80, 255, 120), 2, cv2.LINE_AA)
    cv2.putText(image_bgr, conf_text, (12, 74),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (220, 220, 220), 1, cv2.LINE_AA)

    # Top 3 candidate mini-bars
    sorted_indices = np.argsort(probabilities)[::-1][:3]
    bar_start_x = int(w * 0.50)
    bar_max_w = int(w * 0.46)
    short_names = ["HELLO", "NO", "YES", "THANK YOU", "HELP", "WATER", "GOOD", "PERFECT", "DOCTOR", "LOVE"]

    for i, idx in enumerate(sorted_indices):
        bar_y = 10 + i * 21
        prob = probabilities[idx]
        name = short_names[idx]

        cv2.putText(image_bgr, f"{name}:", (bar_start_x, bar_y + 13),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.36, (200, 200, 200), 1, cv2.LINE_AA)

        track_x = bar_start_x + 75
        cv2.rectangle(image_bgr, (track_x, bar_y + 2), (track_x + bar_max_w - 95, bar_y + 14), (55, 55, 60), -1)
        fill_w = int(prob * (bar_max_w - 95))
        cv2.rectangle(image_bgr, (track_x, bar_y + 2), (track_x + fill_w, bar_y + 14), GESTURE_COLORS[idx], -1)
        cv2.putText(image_bgr, f"{prob*100:.0f}%", (track_x + bar_max_w - 90, bar_y + 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (240, 240, 240), 1, cv2.LINE_AA)

    return image_bgr


def predict_from_image(image_path, model=None, auto_detect=False, save_result=True,
                       show=False, output_path=None, speak=False):
    if model is None:
        model = get_model()

    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise ValueError(f"Could not read image: {image_path}")

    crop_roi = img_bgr
    if auto_detect:
        detector = HandDetector(static_mode=True)
        _, boxes, _ = detector.find_hands(img_bgr, draw=False)
        if boxes:
            x, y, w, h = boxes[0]
            crop_roi = img_bgr[y:y+h, x:x+w]

    processed_tensor = preprocess_image(crop_roi, img_size=100)
    probabilities = model.predict(processed_tensor, verbose=0)[0]
    pred_idx = int(np.argmax(probabilities))
    confidence = float(probabilities[pred_idx])

    pred_label = GESTURE_LABELS[pred_idx]
    spoken_phrase = DEAF_SPOKEN_PHRASES[pred_idx]
    class_code = CLASS_NAMES[pred_idx]

    print("\n" + "=" * 65)
    print("  DEAFVOICE AI: DEAF SIGN LANGUAGE TRANSLATION")
    print("=" * 65)
    print(f"  Input Sign Image   : {os.path.basename(image_path)}")
    print(f"  Recognized Sign    : {pred_label}")
    print(f"  Spoken Voice Text  : \"{spoken_phrase}\"")
    print(f"  Confidence Score   : {confidence * 100:.2f} %")
    print(f"  Raw Dataset Class  : {class_code} (Sign {pred_idx + 1}/10)")
    print("-" * 65)
    print("  Top 3 Sign Candidates:")
    top3 = np.argsort(probabilities)[::-1][:3]
    for rank, idx in enumerate(top3, 1):
        print(f"    {rank}. {GESTURE_LABELS[idx]:<30}: {probabilities[idx]*100:.2f}% -> \"{DEAF_SPOKEN_PHRASES[idx]}\"")
    print("=" * 65)

    annotated = draw_prediction_card(img_bgr.copy(), pred_idx, confidence, probabilities)

    if save_result:
        if output_path is None:
            os.makedirs('results', exist_ok=True)
            fname = os.path.basename(image_path).rsplit('.', 1)[0]
            output_path = os.path.join('results', f"pred_{fname}.jpg")
        cv2.imwrite(output_path, annotated)
        print(f"[INFO] Annotated DeafVoice card saved -> {output_path}")

    if speak:
        synth = AsyncVoiceSynthesizer(enabled=True)
        synth.speak(spoken_phrase)
        import time
        time.sleep(1.0)
        synth.stop()

    if show:
        cv2.imshow("DeafVoice AI -- Sign Translation", annotated)
        print("[INFO] Press any key on preview window to close.")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return pred_label, confidence, probabilities


def predict_directory(dir_path, model=None, auto_detect=False):
    if model is None:
        model = get_model()

    images = [f for f in sorted(os.listdir(dir_path)) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    print(f"\n[INFO] Found {len(images)} sign images in {dir_path}")

    results = []
    for f in images:
        p = os.path.join(dir_path, f)
        label, conf, _ = predict_from_image(p, model=model, auto_detect=auto_detect, save_result=True, show=False)
        results.append((f, label, conf))

    print("\n" + "=" * 70)
    print(f"  DEAFVOICE AI: BATCH TRANSLATION SUMMARY ({len(results)} images)")
    print("=" * 70)
    for fname, label, conf in results:
        idx = [i for i, g in enumerate(GESTURE_LABELS) if g == label][0]
        spoken = DEAF_SPOKEN_PHRASES[idx]
        print(f"  {fname:<25} -> {label:<25} | Voice: \"{spoken}\" ({conf*100:.1f}%)")
    print("=" * 70)
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="DeafVoice AI: Sign Language Image Inference")
    parser.add_argument('--image', type=str, help="Path to single sign image")
    parser.add_argument('--dir', type=str, help="Path to directory of sign images")
    parser.add_argument('--auto_detect', action='store_true', help="Auto-crop hand using MediaPipe")
    parser.add_argument('--show', action='store_true', help="Display preview window")
    parser.add_argument('--speak', action='store_true', help="Speak translated phrase aloud via TTS")
    parser.add_argument('--output', type=str, default=None, help="Output image path")

    args = parser.parse_args()

    if args.image:
        predict_from_image(args.image, auto_detect=args.auto_detect, show=args.show,
                           output_path=args.output, speak=args.speak)
    elif args.dir:
        predict_directory(args.dir, auto_detect=args.auto_detect)
    else:
        sample = 'sample_images/posture_a_sample.jpg'
        if os.path.exists(sample):
            predict_from_image(sample, auto_detect=False, show=False, speak=False)
        else:
            parser.print_help()
