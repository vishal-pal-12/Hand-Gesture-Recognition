# realtime_detect.py -- DeafVoice AI: Real-Time Sign-to-Speech Assistive Communicator
import os
import sys

# Auto-redirect to project venv python if not already running in it
project_root = os.path.dirname(os.path.abspath(__file__))
venv_py = os.path.join(project_root, 'venv', 'Scripts', 'python.exe')
if os.path.exists(venv_py) and os.path.normcase(sys.executable) != os.path.normcase(venv_py):
    import subprocess
    res = subprocess.run([venv_py] + sys.argv, cwd=project_root)
    sys.exit(res.returncode)

import time
import argparse
import cv2
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
from utils.assistive_comm import (
    AsyncVoiceSynthesizer,
    GestureStabilityTracker,
    SentenceBuilder,
    ConversationLogger,
    VOCABULARY_MODES,
    DEAF_DAILY_PHRASES,
    DEAF_ALPHABET,
    DEAF_EMERGENCY_PHRASES
)

MODEL_PATH_PRIMARY = 'models/best_model.keras'
FALLBACK_MODEL_PATH = 'models/hand_posture_cnn_final.keras'


def get_model():
    if os.path.exists(MODEL_PATH_PRIMARY):
        return safe_load_model(MODEL_PATH_PRIMARY)
    elif os.path.exists(FALLBACK_MODEL_PATH):
        return safe_load_model(FALLBACK_MODEL_PATH)
    else:
        raise FileNotFoundError(
            "No trained model found. Please run 'python run.py train' first."
        )


def draw_styled_box(frame, x, y, w, h, label="", color=(0, 255, 128)):
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 1)
    line_len = int(min(w, h) * 0.18)
    thick = 3
    # Top-left
    cv2.line(frame, (x, y), (x + line_len, y), color, thick)
    cv2.line(frame, (x, y), (x, y + line_len), color, thick)
    # Top-right
    cv2.line(frame, (x + w, y), (x + w - line_len, y), color, thick)
    cv2.line(frame, (x + w, y), (x + w, y + line_len), color, thick)
    # Bottom-left
    cv2.line(frame, (x, y + h), (x + line_len, y + h), color, thick)
    cv2.line(frame, (x, y + h), (x, y + h - line_len), color, thick)
    # Bottom-right
    cv2.line(frame, (x + w, y + h), (x + w - line_len, y + h), color, thick)
    cv2.line(frame, (x + w, y + h), (x + w, y + h - line_len), color, thick)

    if label:
        cv2.putText(frame, label, (x, max(20, y - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, color, 2, cv2.LINE_AA)


def draw_deaf_hud(frame, current_sign, current_speech, confidence, hold_progress,
                  sentence_builder, voice_synth, fps, mode_str, show_bars=True,
                  probabilities=None):
    h, w, _ = frame.shape

    # 1. TOP ACCESSIBILITY BANNER (Height: 80px)
    top_h = 80
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, top_h), (20, 20, 25), -1)
    cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)

    # Title
    cv2.putText(frame, "DEAFVOICE AI: SIGN-TO-SPEECH COMMUNICATOR", (15, 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 240, 255), 2, cv2.LINE_AA)

    # Status pills
    tts_status = "TTS: ON [V]" if voice_synth.enabled else "TTS: MUTED [V]"
    tts_color = (80, 255, 80) if voice_synth.enabled else (120, 120, 255)
    cv2.putText(frame, f"{tts_status} | {mode_str} | FPS: {fps:.1f}", (w - 360, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, tts_color, 1, cv2.LINE_AA)

    # Active recognized sign and spoken translation preview
    if current_sign:
        sign_display = f"SIGN: {current_sign} ({confidence*100:.0f}%)"
        speech_display = f'VOICE: "{current_speech}"'
        cv2.putText(frame, sign_display, (15, 53),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, speech_display, (15, 72),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 255, 200), 1, cv2.LINE_AA)
    else:
        cv2.putText(frame, "Waiting for hand sign...", (15, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (160, 160, 160), 1, cv2.LINE_AA)

    # Hold-to-confirm progress gauge in top bar
    gauge_x = w - 360
    gauge_y = 48
    gauge_w = 180
    gauge_h = 16
    cv2.rectangle(frame, (gauge_x, gauge_y), (gauge_x + gauge_w, gauge_y + gauge_h), (60, 60, 60), -1)
    fill_w = int(hold_progress * gauge_w)
    gauge_col = (0, 220, 255) if hold_progress < 0.99 else (0, 255, 0)
    if fill_w > 0:
        cv2.rectangle(frame, (gauge_x, gauge_y), (gauge_x + fill_w, gauge_y + gauge_h), gauge_col, -1)
    cv2.putText(frame, f"Hold: {int(hold_progress*100)}%", (gauge_x + gauge_w + 10, gauge_y + 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.40, (220, 220, 220), 1, cv2.LINE_AA)

    # VISUAL SOUND GLOW (Crucial for Deaf Users who cannot hear audio feedback)
    now = time.time()
    time_since_speech = now - voice_synth.last_spoken_time
    is_speaking = time_since_speech < 1.2
    if is_speaking or sentence_builder.is_flashing():
        cv2.rectangle(frame, (0, 0), (w, top_h), (0, 255, 0), 3)
        pulse_text = "))) VOICE SPOKEN ((("
        cv2.putText(frame, pulse_text, (gauge_x, gauge_y + 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.46, (0, 255, 0), 2, cv2.LINE_AA)

    # 2. BOTTOM ACCESSIBILITY SUBTITLE BOARD (Height: 110px)
    bot_h = 110
    overlay_bot = frame.copy()
    cv2.rectangle(overlay_bot, (0, h - bot_h), (w, h), (15, 15, 20), -1)
    cv2.addWeighted(overlay_bot, 0.88, frame, 0.12, 0, frame)

    # Active Sentence constructed by Deaf User
    current_sentence = sentence_builder.get_sentence()
    if not current_sentence:
        current_sentence = "(Hold signs to build your message...)"
    cv2.putText(frame, "DEAF BOARD :", (15, h - bot_h + 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.46, (0, 200, 255), 1, cv2.LINE_AA)
    cv2.putText(frame, current_sentence, (150, h - bot_h + 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 2, cv2.LINE_AA)

    # Hearing Partner Response (Two-Way Communication)
    reply_text = sentence_builder.hearing_reply
    if not reply_text:
        reply_text = "[R] Hearing partner press 'R' to reply"
        reply_color = (130, 130, 130)
    else:
        reply_color = (0, 255, 255)
    cv2.putText(frame, "HEARING    :", (15, h - bot_h + 56),
                cv2.FONT_HERSHEY_SIMPLEX, 0.46, (255, 200, 0), 1, cv2.LINE_AA)
    cv2.putText(frame, reply_text, (150, h - bot_h + 56),
                cv2.FONT_HERSHEY_SIMPLEX, 0.56, reply_color, 2, cv2.LINE_AA)

    # Navigation / Short-cuts footer
    footer_text = "[TAB] Mode | [Space] Gap | [B] Backspace | [C] Clear | [S] Speak | [R] Reply | [V] TTS | [M] Box | [Q] Quit"
    cv2.putText(frame, footer_text, (15, h - 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180, 180, 180), 1, cv2.LINE_AA)

    # 3. SIDEBAR PROBABILITY BARS (Right side)
    if show_bars and probabilities is not None:
        draw_deaf_probability_bars(frame, probabilities)

    return frame


def draw_deaf_probability_bars(frame, probabilities):
    h, w, _ = frame.shape
    panel_w = 210
    panel_h = 240
    start_x = w - panel_w - 10
    start_y = 86

    overlay = frame.copy()
    cv2.rectangle(overlay, (start_x, start_y), (start_x + panel_w, start_y + panel_h), (18, 18, 22), -1)
    cv2.addWeighted(overlay, 0.80, frame, 0.20, 0, frame)

    cv2.putText(frame, "Sign Probabilities:", (start_x + 8, start_y + 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 240, 255), 1, cv2.LINE_AA)

    short_names = [
        "HELLO", "NO", "YES", "THANK YOU", "HELP",
        "WATER", "GOOD", "PERFECT", "DOCTOR", "LOVE"
    ]

    bar_max_w = 80
    bar_h = 13
    gap = 21

    for i in range(10):
        y_pos = start_y + 35 + i * gap
        prob = probabilities[i]
        label = f"{short_names[i]:<9}"
        color = GESTURE_COLORS[i]

        cv2.putText(frame, label, (start_x + 6, y_pos + 11),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.33, (220, 220, 220), 1, cv2.LINE_AA)

        track_x = start_x + 72
        cv2.rectangle(frame, (track_x, y_pos), (track_x + bar_max_w, y_pos + bar_h), (45, 45, 50), -1)
        fill_w = int(prob * bar_max_w)
        cv2.rectangle(frame, (track_x, y_pos), (track_x + fill_w, y_pos + bar_h), color, -1)
        cv2.putText(frame, f"{prob*100:.0f}%", (track_x + bar_max_w + 4, y_pos + 11),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.33, (240, 240, 240), 1, cv2.LINE_AA)


def run_realtime(camera_index=0, max_frames=None, output_path=None, test_mode=False, speech_enabled=True):
    print("=" * 68)
    print("  DEAFVOICE AI: REAL-TIME SIGN-TO-SPEECH COMMUNICATOR")
    print("  Empowering Deaf & Speech-Impaired Individuals with Sign AI")
    print(f"  Camera Device Index : {camera_index}")
    print(f"  Voice Synthesis TTS : {'ENABLED (Windows SAPI)' if speech_enabled else 'MUTED'}")
    print("=" * 68)

    model = get_model()
    detector = HandDetector()
    voice_synth = AsyncVoiceSynthesizer(enabled=speech_enabled)
    stability_tracker = GestureStabilityTracker(required_frames=12, cooldown_seconds=1.2)
    sentence_builder = SentenceBuilder()
    conv_logger = ConversationLogger()

    print("[OK] DeafVoice components initialized successfully.")
    print("     - Real-Time Subtitles & Communication Board active")
    print("     - Hold-to-Confirm Gesture Stabilization active (~0.5s)")
    print("     - Visual Voice-Glow Pulse active for deaf feedback")
    print("     - Two-Way Hearing Partner Reply active ('R' key)")
    print("-" * 68)

    if test_mode:
        print("[INFO] Running in headless test mode with synthetic frame.")
        synthetic_frame = np.ones((480, 640, 3), dtype=np.uint8) * 110
        cv2.rectangle(synthetic_frame, (200, 100), (450, 350), (200, 180, 160), -1)
        crop = synthetic_frame[100:350, 200:450]
        tensor = preprocess_image(crop, img_size=100)
        probs = model.predict(tensor, verbose=0)[0]
        idx = int(np.argmax(probs))

        sign_name = GESTURE_LABELS[idx]
        spoken_text = DEAF_SPOKEN_PHRASES[idx]
        sentence_builder.add_gesture(idx)

        annotated = draw_deaf_hud(
            synthetic_frame.copy(),
            current_sign=sign_name,
            current_speech=spoken_text,
            confidence=float(probs[idx]),
            hold_progress=1.0,
            sentence_builder=sentence_builder,
            voice_synth=voice_synth,
            fps=30.0,
            mode_str="TestMode",
            show_bars=True,
            probabilities=probs
        )
        if output_path:
            os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
            cv2.imwrite(output_path, annotated)
        print(f"[SUCCESS] Realtime DeafVoice simulation passed. Detected: {sign_name} -> \"{spoken_text}\"")
        voice_synth.stop()
        return

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"[ERROR] Could not open camera {camera_index}.")
        print("  Running headless simulation test instead ...")
        run_realtime(test_mode=True, output_path=output_path, speech_enabled=speech_enabled)
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    auto_detect_mode = detector.has_mediapipe
    show_bars = True
    frame_count = 0
    fps = 0.0
    prev_time = time.time()

    curr_sign = ""
    curr_speech = ""
    curr_conf = 0.0
    curr_probs = np.zeros(10, dtype=np.float32)
    hold_progress = 0.0

    writer = None
    if output_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_path, fourcc, 20.0, (640, 480))

    print("\n[READY] DeafVoice Camera Feed is LIVE! Press 'Q' to quit.\n")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[WARNING] Empty frame received from webcam. Exiting loop.")
                break

            frame = cv2.flip(frame, 1)
            frame_count += 1

            now = time.time()
            fps = 0.9 * fps + 0.1 * (1.0 / max(now - prev_time, 1e-4))
            prev_time = now

            hand_crop = None
            detected_box = None
            mode_display = "MediaPipe Hand Tracking" if auto_detect_mode else "Fixed ROI Guide"

            if auto_detect_mode and detector.has_mediapipe:
                frame, boxes, _ = detector.find_hands(frame, draw=True)
                if boxes:
                    x, y, w_b, h_b = boxes[0]
                    detected_box = (x, y, w_b, h_b)
                    draw_styled_box(frame, x, y, w_b, h_b, "Deaf Sign Input", (0, 255, 128))
                    hand_crop = frame[y:y+h_b, x:x+w_b]
                else:
                    cv2.putText(frame, "Show sign inside camera view",
                                (int(frame.shape[1]*0.25), int(frame.shape[0]*0.50)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.70, (0, 180, 255), 2, cv2.LINE_AA)
            else:
                x, y, w_b, h_b = HandDetector.get_fixed_roi(frame, size=240)
                detected_box = (x, y, w_b, h_b)
                draw_styled_box(frame, x, y, w_b, h_b, "Place Sign Here", (0, 255, 0))
                hand_crop = frame[y:y+h_b, x:x+w_b]

            detected_idx = None
            if hand_crop is not None and hand_crop.size > 0 and hand_crop.shape[0] > 10 and hand_crop.shape[1] > 10:
                tensor = preprocess_image(hand_crop, img_size=100)
                curr_probs = model.predict(tensor, verbose=0)[0]
                detected_idx = int(np.argmax(curr_probs))
                curr_conf = float(curr_probs[detected_idx])
                curr_sign = GESTURE_LABELS[detected_idx]
                curr_speech = DEAF_SPOKEN_PHRASES[detected_idx]

            # Gesture stabilization & hold-to-confirm
            committed_idx, hold_progress = stability_tracker.update(detected_idx, curr_conf, min_conf=0.45)

            if committed_idx is not None:
                token = sentence_builder.add_gesture(committed_idx)
                spoken_phrase = DEAF_SPOKEN_PHRASES[committed_idx]
                voice_synth.speak(spoken_phrase)
                conv_logger.log_entry("Deaf User", f"{GESTURE_LABELS[committed_idx]} -> {spoken_phrase}")
                print(f"[DEAF SPOKEN] \"{spoken_phrase}\" (Sign: {GESTURE_LABELS[committed_idx]}, Conf: {curr_conf*100:.1f}%)")

            display_frame = draw_deaf_hud(
                frame=frame,
                current_sign=curr_sign,
                current_speech=curr_speech,
                confidence=curr_conf,
                hold_progress=hold_progress,
                sentence_builder=sentence_builder,
                voice_synth=voice_synth,
                fps=fps,
                mode_str=sentence_builder.get_mode_name(),
                show_bars=show_bars,
                probabilities=curr_probs
            )

            if writer is not None:
                writer.write(display_frame)

            cv2.imshow("DeafVoice AI -- Sign-to-Speech Communicator", display_frame)

            key = cv2.waitKey(1) & 0xFF
            if key in [ord('q'), 27]:  # 'q' or ESC
                print("[INFO] Closing DeafVoice camera...")
                break
            elif key == ord('\t'):  # TAB key cycles vocab mode
                new_mode = sentence_builder.cycle_mode()
                print(f"[MODE] Switched vocabulary mode -> {new_mode}")
            elif key == ord(' '):  # SPACE key
                sentence_builder.add_space()
                print("[BOARD] Added space.")
            elif key in [8, ord('b')]:  # Backspace or 'b'
                sentence_builder.backspace()
                print("[BOARD] Backspace.")
            elif key == ord('c'):  # Clear board
                sentence_builder.clear()
                print("[BOARD] Cleared conversation board.")
            elif key == ord('s'):  # Speak current board aloud
                full_sent = sentence_builder.get_sentence()
                if full_sent:
                    voice_synth.speak(full_sent, cooldown=0.2)
                    print(f"[SPEAKING FULL BOARD] \"{full_sent}\"")
            elif key == ord('r'):  # Hearing partner reply
                print("\n" + "=" * 55)
                print("  HEARING PARTNER REPLY INPUT (Type below & press Enter)")
                print("=" * 55)
                try:
                    reply = input("  Hearing Reply: ")
                    if reply.strip():
                        sentence_builder.set_hearing_reply(reply)
                        conv_logger.log_entry("Hearing Partner", reply)
                        print(f"  [SENT TO DEAF SCREEN] -> \"{reply}\"\n")
                except Exception as e:
                    print(f"  [WARNING] Could not read reply: {e}")
            elif key == ord('v'):  # Toggle TTS voice
                voice_synth.enabled = not voice_synth.enabled
                print(f"[TTS VOICE] Voice synthesis is now {'ENABLED' if voice_synth.enabled else 'MUTED'}")
            elif key == ord('m'):  # Toggle Hand Detection
                auto_detect_mode = not auto_detect_mode
                print(f"[DETECTION] Mode switched to: {'MediaPipe' if auto_detect_mode else 'Fixed ROI'}")
            elif key == ord('p'):  # Toggle probability bars
                show_bars = not show_bars
                print(f"[BARS] Probability panel: {show_bars}")
            elif key == ord('x'):  # Save snapshot
                shot_path = f"results/deafvoice_snapshot_{int(time.time())}.jpg"
                os.makedirs('results', exist_ok=True)
                cv2.imwrite(shot_path, display_frame)
                print(f"[INFO] Saved snapshot -> {shot_path}")

            if max_frames and frame_count >= max_frames:
                print(f"[INFO] Reached max frames ({max_frames}).")
                break

    finally:
        voice_synth.stop()
        cap.release()
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()
        print("[INFO] Camera released. DeafVoice session ended.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="DeafVoice AI: Live Sign-to-Speech Assistive Communicator for Deaf & Mute"
    )
    parser.add_argument('--camera', type=int, default=0, help="Camera device index (default: 0)")
    parser.add_argument('--max_frames', type=int, default=None, help="Max frames to process")
    parser.add_argument('--save_output', type=str, default=None, help="Save video path")
    parser.add_argument('--test_mode', action='store_true', help="Run headless test mode")
    parser.add_argument('--mute', action='store_true', help="Disable voice TTS")

    args = parser.parse_args()
    run_realtime(
        camera_index=args.camera,
        max_frames=args.max_frames,
        output_path=args.save_output,
        test_mode=args.test_mode,
        speech_enabled=not args.mute
    )
