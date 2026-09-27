"""
============================================================
  utils/assistive_comm.py
  DeafVoice AI -- Assistive Communication Engine
  Dedicated to Deaf & Speech-Impaired Individuals
============================================================
"""

import os
import time
import queue
import threading
from datetime import datetime

# Mode 1: Daily Essential Communication Phrases
DEAF_DAILY_PHRASES = {
    0: "Hello, Welcome",
    1: "No, Stop",
    2: "Yes, I Agree",
    3: "Thank You",
    4: "I Need Help",
    5: "Water Please",
    6: "Good, I Understand",
    7: "All Good, Perfect",
    8: "Need Doctor or Medicine",
    9: "I Love You, Goodbye"
}

# Mode 2: Fingerspelling Alphabet (A to J)
DEAF_ALPHABET = {
    0: "A",
    1: "B",
    2: "C",
    3: "D",
    4: "E",
    5: "F",
    6: "G",
    7: "H",
    8: "I",
    9: "J"
}

# Mode 3: Emergency / Medical Signs
DEAF_EMERGENCY_PHRASES = {
    0: "Attention Please",
    1: "Emergency Stop",
    2: "Yes, Urgent",
    3: "Need Help Immediately",
    4: "Severe Pain",
    5: "Need Water / Thirsty",
    6: "I Am Okay Now",
    7: "Please Call An Ambulance",
    8: "Call A Doctor Right Now",
    9: "Contact My Family"
}

VOCABULARY_MODES = [
    "Daily Essentials",
    "Fingerspelling (A-J)",
    "Emergency / Medical"
]


class AsyncVoiceSynthesizer:
    """Non-blocking text-to-speech engine using Windows SAPI in background thread."""
    def __init__(self, enabled=True, speech_rate=1, volume=100):
        self.enabled = enabled
        self.speech_rate = speech_rate
        self.volume = volume
        self.speech_queue = queue.Queue()
        self.is_running = True
        self.last_spoken_text = ""
        self.last_spoken_time = 0.0

        self.worker_thread = threading.Thread(target=self._speech_worker, daemon=True)
        self.worker_thread.start()

    def _speech_worker(self):
        speaker = None
        try:
            import pythoncom
            pythoncom.CoInitialize()
            import win32com.client
            speaker = win32com.client.Dispatch("SAPI.SpVoice")
            speaker.Rate = self.speech_rate
            speaker.Volume = self.volume
        except Exception:
            speaker = None

        while self.is_running:
            try:
                text = self.speech_queue.get(timeout=0.2)
                if text and self.enabled:
                    self.last_spoken_text = text
                    self.last_spoken_time = time.time()
                    if speaker:
                        try:
                            speaker.Speak(text)
                        except Exception:
                            pass
                self.speech_queue.task_done()
            except queue.Empty:
                continue
            except Exception:
                pass

    def speak(self, text, cooldown=1.2):
        if not self.enabled or not text.strip():
            return
        now = time.time()
        if text == self.last_spoken_text and (now - self.last_spoken_time) < cooldown:
            return
        self.speech_queue.put(text.strip())

    def stop(self):
        self.is_running = False


class GestureStabilityTracker:
    """Requires holding a gesture steadily for required_frames to confirm."""
    def __init__(self, required_frames=12, cooldown_seconds=1.2):
        self.required_frames = required_frames
        self.cooldown_seconds = cooldown_seconds
        self.current_candidate = None
        self.candidate_count = 0
        self.last_committed_gesture = None
        self.last_commit_time = 0.0

    def update(self, detected_idx, confidence, min_conf=0.45):
        now = time.time()
        if detected_idx is None or confidence < min_conf:
            self.candidate_count = max(0, self.candidate_count - 1)
            return None, 0.0

        if detected_idx == self.current_candidate:
            self.candidate_count += 1
        else:
            self.current_candidate = detected_idx
            self.candidate_count = 1

        progress = min(1.0, self.candidate_count / self.required_frames)

        if self.candidate_count >= self.required_frames:
            if (detected_idx != self.last_committed_gesture) or (now - self.last_commit_time > self.cooldown_seconds):
                self.last_committed_gesture = detected_idx
                self.last_commit_time = now
                self.candidate_count = 0
                return detected_idx, 1.0

        return None, progress


class SentenceBuilder:
    """Maintains the live sentence constructed by the deaf user."""
    def __init__(self):
        self.tokens = []
        self.active_mode = 0
        self.flash_visual_until = 0.0
        self.hearing_reply = ""
        self.hearing_reply_timestamp = 0.0

    def add_gesture(self, gesture_idx):
        if self.active_mode == 0:
            token = DEAF_DAILY_PHRASES.get(gesture_idx, "")
            if token:
                self.tokens.append(token)
                self.flash()
                return token
        elif self.active_mode == 1:
            letter = DEAF_ALPHABET.get(gesture_idx, "")
            if letter:
                self.tokens.append(letter)
                self.flash()
                return letter
        elif self.active_mode == 2:
            urgent_msg = DEAF_EMERGENCY_PHRASES.get(gesture_idx, "")
            if urgent_msg:
                self.tokens.append(urgent_msg)
                self.flash()
                return urgent_msg
        return ""

    def add_space(self):
        self.tokens.append(" ")

    def backspace(self):
        if self.tokens:
            self.tokens.pop()

    def clear(self):
        self.tokens.clear()

    def cycle_mode(self):
        self.active_mode = (self.active_mode + 1) % len(VOCABULARY_MODES)
        return VOCABULARY_MODES[self.active_mode]

    def get_mode_name(self):
        return VOCABULARY_MODES[self.active_mode]

    def get_sentence(self):
        if self.active_mode == 1:
            return "".join(self.tokens)
        else:
            return " | ".join(self.tokens)

    def flash(self, duration=0.6):
        self.flash_visual_until = time.time() + duration

    def is_flashing(self):
        return time.time() < self.flash_visual_until

    def set_hearing_reply(self, text):
        self.hearing_reply = text.strip()
        self.hearing_reply_timestamp = time.time()


class ConversationLogger:
    """Persists conversations between deaf user and hearing user to text log."""
    def __init__(self, log_path='results/conversation_transcript.txt'):
        self.log_path = log_path
        os.makedirs(os.path.dirname(log_path), exist_ok=True)

    def log_entry(self, speaker, message):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        entry = f"[{timestamp}] {speaker}: {message}\n"
        with open(self.log_path, 'a', encoding='utf-8') as f:
            f.write(entry)
