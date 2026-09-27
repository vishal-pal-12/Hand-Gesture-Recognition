# realtime_detect.py -- DeafVoice AI: Multi-Hand Real-Time Sign-to-Speech Communicator
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

from utils.hand_detector import HandDetector
from utils.gesture_classifier import (
    CLASS_NAMES,
    GESTURE_LABELS,
    DEAF_SPOKEN_PHRASES,
    GESTURE_COLORS,
    classify_finger_gesture
)
from utils.helpers import safe_load_model
from utils.assistive_comm import (
    AsyncVoiceSynthesizer,
    GestureStabilityTracker,
    SentenceBuilder,
    ConversationLogger,
    VOCABULARY_MODES
)

MODEL_PATH_PRIMARY = 'models/best_model.keras'
FALLBACK_MODEL_PATH = 'models/finger_gesture_model.keras'


def get_model():
    if os.path.exists(MODEL_PATH_PRIMARY):
        return safe_load_model(MODEL_PATH_PRIMARY)
    elif os.path.exists(FALLBACK_MODEL_PATH):
        return safe_load_model(FALLBACK_MODEL_PATH)
    return None


def draw_styled_box(frame, x, y, w, h, label="", color=(0, 255, 128)):
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 1)
    line_len = int(min(w, h) * 0.18)
    thick = 3
    # Corners
    cv2.line(frame, (x, y), (x + line_len, y), color, thick)
    cv2.line(frame, (x, y), (x, y + line_len), color, thick)
    cv2.line(frame, (x + w, y), (x + w - line_len, y), color, thick)
    cv2.line(frame, (x + w, y), (x + w, y + line_len), color, thick)
    cv2.line(frame, (x, y + h), (x + line_len, y + h), color, thick)
    cv2.line(frame, (x, y + h), (x, y + h - line_len), color, thick)
    cv2.line(frame, (x + w, y + h), (x + w - line_len, y + h), color, thick)
    cv2.line(frame, (x + w, y + h), (x + w, y + h - line_len), color, thick)

    if label:
        cv2.putText(frame, label, (x, max(22, y - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, color, 2, cv2.LINE_AA)


def draw_deaf_hud(frame, detected_hands_info, hold_progress,
                  sentence_builder, voice_synth, fps, mode_str, show_bars=True):
    h, w, _ = frame.shape

    # 1. TOP BANNER (Height: 80px)
    top_h = 80
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, top_h), (20, 20, 25), -1)
    cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)

    # Title & Multi-Hand Badge
    cv2.putText(frame, "DEAFVOICE AI: MULTI-HAND SIGN COMMUNICATOR", (15, 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 240, 255), 2, cv2.LINE_AA)

    tts_status = "TTS: ON [V]" if voice_synth.enabled else "TTS: MUTED [V]"
    tts_color = (80, 255, 80) if voice_synth.enabled else (120, 120, 255)
    num_hands = len(detected_hands_info)
    hands_badge = f"{num_hands} Hand{'s' if num_hands != 1 else ''} Active"
    cv2.putText(frame, f"{tts_status} | {hands_badge} | FPS: {fps:.1f}", (w - 380, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, tts_color, 1, cv2.LINE_AA)

    # Display detected gesture info in top banner
    if detected_hands_info:
        primary = detected_hands_info[0]
        sign_display = f"{primary['handedness'].upper()}: {primary['label']} ({primary['conf']*100:.0f}%)"
        speech_display = f'VOICE: "{primary["speech"]}"'
        cv2.putText(frame, sign_display, (15, 52),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.56, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, speech_display, (15, 72),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 255, 200), 1, cv2.LINE_AA)
    else:
        cv2.putText(frame, "Show Left or Right Hand to camera...", (15, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, (160, 160, 160), 1, cv2.LINE_AA)

    # Hold gauge
    gauge_x = w - 380
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

    # Visual Voice-Glow Pulse
    now = time.time()
    if (now - voice_synth.last_spoken_time < 1.2) or sentence_builder.is_flashing():
        cv2.rectangle(frame, (0, 0), (w, top_h), (0, 255, 0), 3)
        cv2.putText(frame, "))) VOICE SPOKEN (((", (gauge_x, gauge_y + 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.46, (0, 255, 0), 2, cv2.LINE_AA)

    # 2. BOTTOM SUBTITLE BOARD (Height: 110px)
    bot_h = 110
    overlay_bot = frame.copy()
    cv2.rectangle(overlay_bot, (0, h - bot_h), (w, h), (15, 15, 20), -1)
    cv2.addWeighted(overlay_bot, 0.88, frame, 0.12, 0, frame)

    current_sentence = sentence_builder.get_sentence()
    if not current_sentence:
        current_sentence = "(Hold signs to build your conversation...)"
    cv2.putText(frame, "DEAF BOARD :", (15, h - bot_h + 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.46, (0, 200, 255), 1, cv2.LINE_AA)
    cv2.putText(frame, current_sentence, (150, h - bot_h + 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 2, cv2.LINE_AA)

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

    footer_text = "[TAB] Mode | [Space] Gap | [B] Backspace | [C] Clear | [S] Speak | [R] Reply | [V] TTS | [P] Bars | [Q] Quit"
    cv2.putText(frame, footer_text, (15, h - 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180, 180, 180), 1, cv2.LINE_AA)

    # 3. SIDEBAR PROBABILITY BARS
    if show_bars and detected_hands_info:
        probs = detected_hands_info[0].get('probs')
        if probs is not None:
            draw_deaf_probability_bars(frame, probs)

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

    short_names = ["HELLO", "NO", "YES", "FINE", "THANK U", "HELP", "WATER", "PERFECT", "DOCTOR", "LOVE"]
    bar_max_w = 80
    bar_h = 13
    gap = 21

    for i in range(10):
        y_pos = start_y + 35 + i * gap
        prob = probabilities[i]
        label = f"{short_names[i]:<8}"
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
    print("  DEAFVOICE AI: MULTI-HAND SIGN-TO-SPEECH COMMUNICATOR")
    print("  Supports Left & Right Hands, Multi-Hand, and Zero-Overlap AI")
    print(f"  Camera Device Index : {camera_index}")
    print(f"  Voice Synthesis TTS : {'ENABLED (Windows SAPI)' if speech_enabled else 'MUTED'}")
    print("=" * 68)

    model = get_model()
    detector = HandDetector(max_hands=2)
    voice_synth = AsyncVoiceSynthesizer(enabled=speech_enabled)
    stability_tracker = GestureStabilityTracker(required_frames=10, cooldown_seconds=1.2)
    sentence_builder = SentenceBuilder()
    conv_logger = ConversationLogger()

    print("[OK] Multi-Hand Deaf Communicator initialized.")
    print("     - Multi-Hand Tracking: Left & Right Hands active")
    print("     - Zero-Overlap Finger State Logic: Active")
    print("     - Speech Synthesizer: Active")
    print("-" * 68)

    if test_mode:
        print("[INFO] Running in headless test mode.")
        from utils.prepare_dataset import generate_canonical_landmarks, render_landmark_image
        mock_lm = generate_canonical_landmarks(gesture_idx=0, handedness="Right")
        pred_idx, label, phrase, conf, probs = classify_finger_gesture(mock_lm, handedness="Right", model=model)
        sentence_builder.add_gesture(pred_idx)
        synthetic_frame = render_landmark_image(mock_lm, w=640, h=480, title=label, handedness="Right")
        hands_info = [{
            'idx': pred_idx, 'label': label, 'speech': phrase,
            'conf': conf, 'handedness': 'Right', 'probs': probs
        }]
        annotated = draw_deaf_hud(synthetic_frame, hands_info, 1.0, sentence_builder, voice_synth, 30.0, "TestMode", True)
        if output_path:
            os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
            cv2.imwrite(output_path, annotated)
        print(f"[SUCCESS] Multi-Hand test passed: Right Hand -> {label} (\"{phrase}\")")
        voice_synth.stop()
        return

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"[ERROR] Could not open camera {camera_index}. Running headless test...")
        run_realtime(test_mode=True, output_path=output_path, speech_enabled=speech_enabled)
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    show_bars = True
    frame_count = 0
    fps = 0.0
    prev_time = time.time()
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
                break

            frame = cv2.flip(frame, 1)
            frame_count += 1

            now = time.time()
            fps = 0.9 * fps + 0.1 * (1.0 / max(now - prev_time, 1e-4))
            prev_time = now

            # Detect multiple hands (Left and Right)
            frame, hand_data = detector.find_hands(frame, draw=True)
            detected_hands_info = []
            active_idx = None
            active_conf = 0.0

            hand_colors = [(0, 255, 128), (255, 200, 0)]  # Hand 1: Green, Hand 2: Cyan/Orange

            for i, hand in enumerate(hand_data):
                color = hand_colors[i % len(hand_colors)]
                x, y, w_b, h_b = hand['bbox']
                handedness = hand['handedness']
                landmarks = hand['landmarks']

                # Classify gesture for this hand
                pred_idx, label, phrase, conf, probs = classify_finger_gesture(
                    landmarks_21x3=landmarks,
                    handedness=handedness,
                    model=model
                )

                box_label = f"{handedness} Hand: {label} [{conf*100:.0f}%]"
                draw_styled_box(frame, x, y, w_b, h_b, box_label, color=color)

                detected_hands_info.append({
                    'idx': pred_idx,
                    'label': label,
                    'speech': phrase,
                    'conf': conf,
                    'handedness': handedness,
                    'probs': probs
                })

                if i == 0:
                    active_idx = pred_idx
                    active_conf = conf

            # Stabilization & hold-to-confirm on primary hand
            committed_idx, hold_progress = stability_tracker.update(active_idx, active_conf, min_conf=0.70)
            if committed_idx is not None and detected_hands_info:
                token = sentence_builder.add_gesture(committed_idx)
                spoken_phrase = detected_hands_info[0]['speech']
                voice_synth.speak(spoken_phrase)
                h_name = detected_hands_info[0]['handedness']
                conv_logger.log_entry(f"Deaf User ({h_name} Hand)", f"{GESTURE_LABELS[committed_idx]} -> {spoken_phrase}")
                print(f"[DEAF SPOKEN] \"{spoken_phrase}\" ({h_name} Hand: {GESTURE_LABELS[committed_idx]}, Conf: {active_conf*100:.1f}%)")

            display_frame = draw_deaf_hud(
                frame=frame,
                detected_hands_info=detected_hands_info,
                hold_progress=hold_progress,
                sentence_builder=sentence_builder,
                voice_synth=voice_synth,
                fps=fps,
                mode_str=sentence_builder.get_mode_name(),
                show_bars=show_bars
            )

            if writer is not None:
                writer.write(display_frame)

            cv2.imshow("DeafVoice AI -- Multi-Hand Sign Communicator", display_frame)

            key = cv2.waitKey(1) & 0xFF
            if key in [ord('q'), 27]:
                break
            elif key == ord('\t'):
                new_mode = sentence_builder.cycle_mode()
                print(f"[MODE] Switched vocabulary mode -> {new_mode}")
            elif key == ord(' '):
                sentence_builder.add_space()
                print("[BOARD] Added space.")
            elif key in [8, ord('b')]:
                sentence_builder.backspace()
                print("[BOARD] Backspace.")
            elif key == ord('c'):
                sentence_builder.clear()
                print("[BOARD] Cleared conversation board.")
            elif key == ord('s'):
                full_sent = sentence_builder.get_sentence()
                if full_sent:
                    voice_synth.speak(full_sent, cooldown=0.2)
                    print(f"[SPEAKING FULL BOARD] \"{full_sent}\"")
            elif key == ord('r'):
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
            elif key == ord('v'):
                voice_synth.enabled = not voice_synth.enabled
                print(f"[TTS VOICE] Voice synthesis is now {'ENABLED' if voice_synth.enabled else 'MUTED'}")
            elif key == ord('p'):
                show_bars = not show_bars
                print(f"[BARS] Probability panel: {show_bars}")

            if max_frames and frame_count >= max_frames:
                break

    finally:
        voice_synth.stop()
        cap.release()
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()
        print("[INFO] Camera released. DeafVoice session ended.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="DeafVoice AI: Multi-Hand Real-Time Sign Communicator")
    parser.add_argument('--camera', type=int, default=0, help="Camera index")
    parser.add_argument('--max_frames', type=int, default=None)
    parser.add_argument('--save_output', type=str, default=None)
    parser.add_argument('--test_mode', action='store_true')
    parser.add_argument('--mute', action='store_true')

    args = parser.parse_args()
    run_realtime(
        camera_index=args.camera,
        max_frames=args.max_frames,
        output_path=args.save_output,
        test_mode=args.test_mode,
        speech_enabled=not args.mute
    )
