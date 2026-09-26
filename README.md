# 🤟 Real-Time Sign Language Translator & Indian Language Speech AI

A state-of-the-art Real-Time Sign Language Recognition & Translation web application built using **Flask, MediaPipe, OpenCV, Keras LSTM Deep Learning**, and **Web Speech API**.

---

## ⚡ Quick Start Guide

### 1. Open Terminal & Navigate to Project Directory
```bash
cd "/Users/jayakrishna/Desktop/MINOR PORJECT/sign language"
```

### 2. Activate the Virtual Environment
```bash
source venv/bin/activate
```

### 3. Start the Flask Server
```bash
python apps.py
```

### 4. Open in Your Browser
Visit:
- **Home Page**: [http://127.0.0.1:8000](http://127.0.0.1:8000)
- **Scanning Dashboard**: [http://127.0.0.1:8000/scan_gesture](http://127.0.0.1:8000/scan_gesture)

---

## 🌟 Key Features

1. **HD Centered Scanning Box**:
   - High-definition (1280x720) video stream with centered dynamic Region of Interest (ROI).
   - Dual-mode support: **Browser WebRTC** (in-browser canvas processing) and **Local OpenCV** (server-side stream).

2. **Letter-to-Word Formation (A–Z Spelling)**:
   - Accumulates individual letter gestures into words in real time (e.g. `H` → `E` → `L` → `L` → `O` = `"HELLO"`).
   - **Auto-Commit**: Resting/dropping your hand for ~0.8s automatically commits the word to the sentence.
   - **Manual Controls**: `Spacebar / Enter` to commit, `Backspace` to delete letter, `Clear` to reset word buffer.
   - **Whole Words/Phrases**: Gestures like `HELLO`, `GOOD AFTERNOON`, `THANK YOU` are recognized as whole phrases.

3. **Multi-Language Indian Translation**:
   - Supports translation from English to **10+ Indian Languages**:
     - 🇮🇳 Kannada (ಕನ್ನಡ)
     - 🇮🇳 Hindi (हिन्दी)
     - 🇮🇳 Tamil (தமிழ்)
     - 🇮🇳 Telugu (తెలుగు)
     - 🇮🇳 Malayalam (മലയാളം)
     - 🇮🇳 Marathi (मराठी)
     - 🇮🇳 Bengali (বাংলা)
     - 🇮🇳 Gujarati (ગુજરાતી)
     - 🇮🇳 Punjabi (ਪੰਜਾਬੀ)
     - 🇮🇳 Odia (ଓଡ଼ିଆ)

4. **Natural Speech Synthesis (TTS)**:
   - Native voice audio generation for recognized English and translated Indian languages.
   - Adjustable speech rate playback.

5. **Premium Dark Glassmorphic Interface**:
   - Designed with modern gradients, glowing letter badges, real-time confidence dials, top-5 probability distribution charts, and keyboard shortcuts.

---

## ⌨️ Keyboard Shortcuts on Scan Dashboard

| Key | Action |
| :--- | :--- |
| **`Space`** or **`Enter`** | Complete / Commit current spelled word to sentence |
| **`Backspace`** | Delete last letter (or last word) |

---

## 👥 Project Team & Credits
- **Institution**: Dayananda Sagar Academy of Technology and Management (DSATM)
- **Project Team**: G Jayakrishna Reddy, Darshan S, G Chandan Babu, Akash A J
