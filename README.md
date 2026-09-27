# DeafVoice AI — Multi-Hand Finger-Wise Assistive Sign Communicator

> **Empowering Deaf, Mute, and Hearing-Impaired Individuals with Real-Time Sign-to-Speech Communication.**  
> **Features**: Multi-Hand Tracking, Left & Right Hand Support, Finger-Wise Zero-Overlap Classification, and Windows SAPI Voice Synthesis.

---

## 1. Zero-Overlap Finger Gesture Vocabulary

Every gesture has a completely unique finger configuration that works with **both Left and Right hands**:

| Gesture | Finger Configuration | Assistive Meaning / Spoken Voice | Function |
|:---|:---|:---|:---|
| **HELLO** | Only **Index Finger** extended UP | *"Hello, Nice to meet you"* | Greeting & Attention |
| **NO** | **Closed Fist** (all 5 curled) | *"No, Please stop"* | Disagreement & Stop |
| **YES** | **Open Palm** (all 5 extended) | *"Yes, I agree and understand"* | Affirmation & Confirmation |
| **FINE / GOOD** | **Thumbs Up** (thumb up, 4 curled) | *"I am fine, everything is good"* | Clarity & Well-being |
| **THANK YOU** | **Peace / V Sign** (Index + Middle open) | *"Thank you very much"* | Gratitude & Politeness |
| **HELP** | **Three Fingers** (Index, Middle, Ring) | *"Please help me, I need assistance"* | Immediate Assistance |
| **WAIT** | **Four Fingers** (Index, Middle, Ring, Pinky) | *"Please wait a moment"* | Pause & Patience |
| **PERFECT** | **OK Sign** (Thumb & Index circle, 3 open) | *"Everything is perfect and all good"* | Satisfaction |
| **DOCTOR / CALL** | **Phone Sign** (Thumb & Pinky open) | *"I need a doctor or call someone"* | Medical & Urgent Call |
| **I LOVE YOU** | **ILY Sign** (Thumb, Index, Pinky open) | *"I love you, Goodbye"* | Affection & Parting |

---

## 2. Multi-Hand & Left/Right Hand Capabilities

1. **Left & Right Hand Symmetry**: Automatically mirrors coordinate representations for Left hands, ensuring identical 100% accuracy regardless of which hand you use.
2. **Multiple Hands in Real-Time**: Tracks up to 2 hands simultaneously (`Left Hand` in green box, `Right Hand` in cyan/orange box) with live predictions displayed for both hands.
3. **Voice Synthesis (TTS)**: Automatically speaks confirmed signs aloud via Windows SAPI so hearing people can listen.
4. **Visual Voice-Glow Pulse**: Neon pulse flashes on screen when voice speaks, providing visual feedback for deaf individuals.
5. **Two-Way Reply (`[R]`)**: Hearing partner can press `[R]` to type a response, which appears in large yellow text on the screen for the deaf user.

---

## 3. Quickstart Guide (VS Code Terminal)

### Step 1: Open VS Code Terminal
```powershell
cd "C:\Users\VISHAL PAL\OneDrive\Desktop\Hand-Gesture-Recognition"
```

### Step 2: Run Comprehensive Multi-Hand Verification
```powershell
python run.py verify
```

### Step 3: Run Unit Tests (7/7 Passed)
```powershell
python test_project.py
```

### Step 4: Test Static Sign Image Translation
```powershell
python run.py predict --image sample_images/index_hello_sample.jpg
python run.py predict --image sample_images/thumbs_fine_sample.jpg
python run.py predict --image sample_images/palm_yes_sample.jpg
python run.py predict --image sample_images/fist_no_sample.jpg
```

### Step 5: Launch Live Multi-Hand Camera Communicator
```powershell
python run.py deaf-assist
```
*(or `python run.py webcam`, or double-click `run.bat`)*

---

## 4. Live Interactive Keyboard Controls

| Key | Action |
|:---:|:---|
| **`[TAB]`** | Switch vocabulary mode (*Daily Needs $\leftrightarrow$ Alphabet $\leftrightarrow$ Emergency*) |
| **`[Space]`** | Add space to message board |
| **`[B]`** | Backspace (undo last sign) |
| **`[C]`** | Clear message board |
| **`[S]`** | Speak entire message board aloud |
| **`[R]`** | **Hearing Person Reply**: Prompts to type a reply that shows on screen for the deaf user |
| **`[V]`** | Toggle TTS Voice (*Mute / Unmute*) |
| **`[P]`** | Toggle Probability panel |
| **`[Q]`** | Quit application |
