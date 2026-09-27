# DeafVoice AI — Assistive Static Hand Gesture & Two-Way Voice Communicator

> **Empowering Deaf, Mute, and Hearing-Impaired Individuals with Real-Time Sign-to-Speech and Visual Conversation.**  
> Built upon the Deep Convolutional Neural Network research by *Adithya V. & Rajesh R.* (*Procedia Computer Science 171 (2020) 2353–2361*).

---

## 1. Project Mission & Problem Solved

Deaf and speech-impaired individuals face massive everyday communication barriers when interacting with hearing people who do not understand sign language. 

**DeafVoice AI** bridges this gap as a complete **two-way assistive communication hub**:
1. **Sign-to-Speech (Deaf $\rightarrow$ Hearing)**: The deaf user performs natural hand postures. The CNN recognizes the gesture, displays the phrase in large accessibility subtitles, and **speaks it out loud** through a non-blocking voice synthesizer so hearing people hear the deaf person's voice clearly.
2. **Speech-to-Text / Reply (Hearing $\rightarrow$ Deaf)**: The hearing conversational partner can reply directly on-screen, which instantly displays as high-contrast captions for the deaf individual.
3. **Visual Confirmation Pulses**: Since deaf individuals cannot hear audio confirmations, the system flashes a gentle neon border whenever a phrase is committed and spoken aloud.
4. **Emergency Quick-Signs**: Instant trigger signs for *Medical Help, Water, Severe Pain, Calling an Ambulance, and Contacting Family*.

---

## 2. Assistive Communication Modes

Press **`T`** at any time during live detection to switch between modes:

### Mode 1: Daily Essential Needs (Everyday Life)
| Hand Gesture | Assistive Meaning / Spoken Voice | Purpose |
|---|---|---|
| **Posture 1 (Open Palm)** | *"Hello, Welcome"* | Greetings & Attention |
| **Posture 2 (Closed Fist)** | *"No, Stop"* | Disagreement & Boundaries |
| **Posture 3 (Index Point)** | *"Yes, I Agree"* | Agreement & Confirmation |
| **Posture 4 (Peace / V)** | *"Thank You"* | Gratitude & Politeness |
| **Posture 5 (Three Fingers)** | *"I Need Help"* | Assistance Request |
| **Posture 6 (Four Fingers)** | *"Water Please"* | Basic Sustenance |
| **Posture 7 (Thumbs Up)** | *"Good, I Understand"* | Affirmation & Clarity |
| **Posture 8 (OK Sign)** | *"All Good, Perfect"* | Satisfaction |
| **Posture 9 (C-Shape)** | *"Need Doctor or Medicine"* | Health & Clinical |
| **Posture 10 (L-Shape)** | *"I Love You, Goodbye"* | Affection & Parting |

---

### Mode 2: Fingerspelling Alphabet (A to J)
Used for letter-by-letter spelling of personal names, place names, numbers, or specific words:
- `Posture 1`: **A** | `Posture 2`: **B** | `Posture 3`: **C** | `Posture 4`: **D** | `Posture 5`: **E**
- `Posture 6`: **F** | `Posture 7`: **G** | `Posture 8`: **H** | `Posture 9`: **I** | `Posture 10`: **J**

---

### Mode 3: Emergency & Medical Signs
Instant priority phrases for clinics, hospitals, or urgent distress:
- *Attention Please*, *Emergency Stop*, *Severe Pain*, *Please Call An Ambulance*, *Contact My Family*.

---

## 3. Architecture & Performance

- **Network**: 3 Convolutional Layers ($19\times19$, $17\times17$, $15\times15$) + Batch Normalization + ReLU + MaxPooling ($2\times2$, stride 3) + Softmax.
- **Model Size**: Only **166,266 parameters (~650 KB)**.
- **Speed**: Runs in real-time at **>30 FPS** on standard CPU.
- **Accuracy**: **85.00% Test Accuracy**, **86.14% Macro Precision**, **84.84% Macro F1-Score** on 400 complex background benchmark images.

---

## 4. Quickstart Guide (VS Code Terminal)

### Step 1: Open VS Code Terminal
```powershell
cd "C:\Users\VISHAL PAL\OneDrive\Desktop\Hand-Gesture-Recognition"
```

### Step 2: Activate Environment
```powershell
.\venv\Scripts\Activate.ps1
```
*(Note: If you run without activation, the scripts automatically detect and self-route to the venv!)*

### Step 3: Run System Verification
```powershell
python run.py verify
```

### Step 4: Run Automated Unit Tests (7/7 Passed)
```powershell
python test_project.py
```

### Step 5: Launch Real-Time DeafVoice Communicator
```powershell
python run.py deaf-assist --camera 0
```
*(or simply `python run.py webcam`)*

---

## 5. Live Interactive Keyboard Controls

| Key | Action |
|---|---|
| **`T`** | Cycle Vocabulary Mode (*Daily Essentials $\rightarrow$ Alphabet $\rightarrow$ Emergency*) |
| **`V`** | Speak the entire constructed sentence aloud |
| **`C`** | Backspace (delete last added word or letter) |
| **`X`** | Clear message board completely |
| **`Space`** | Add space to message |
| **`R`** | **Hearing Person Reply**: Prompts hearing person to type a message that displays in large text on screen for the deaf user |
| **`M`** | Toggle between **MediaPipe Auto-Tracking** and **Fixed ROI Guide Box** |
| **`B`** | Toggle Probability distribution HUD |
| **`S`** | Save conversation transcript & snapshot to `results/` |
| **`Q`** | Quit application |

---

## 6. Citation

```bibtex
@article{adithya2020deep,
  title={A Deep Convolutional Neural Network Approach for Static Hand Gesture Recognition},
  author={Adithya, V. and Rajesh, R.},
  journal={Procedia Computer Science},
  volume={171},
  pages={2353--2361},
  year={2020},
  publisher={Elsevier},
  doi={10.1016/j.procs.2020.04.255}
}
```

