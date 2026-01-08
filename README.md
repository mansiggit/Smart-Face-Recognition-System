# 🤖 Smart Face Recognition System

A high-performance, real-time facial recognition solution built using Deep Learning and Computer Vision. This system processes live video streams to identify individuals by comparing real-time biometric signatures against a pre-encoded database.

## ✨ Key Features
- **Real-time Recognition:** Sub-second identification of individuals via webcam.
- **Deep Learning Core:** Powered by dlib's state-of-the-art face recognition model (99.38% accuracy).
- **Optimized Pipeline:** Uses 0.25x frame scaling to maintain high FPS during live processing.
- **Multi-Face Support:** Capable of detecting and labeling multiple identities simultaneously in a single frame.

## 🛠️ Tech Stack
- **Language:** Python
- **UI Framework:** Streamlit
- **Streaming:** Streamlit-WebRTC (for local browser-based camera access)
- **Computer Vision:** OpenCV
- **Deep Learning Library:** Face-Recognition (dlib)
- **Data Handling:** NumPy & Pickle

## ⚙️ Technical Logic
1. **Preprocessing:** Frames are captured in BGR format, downscaled for speed, and converted to RGB.
2. **Feature Extraction:** The system detects face locations and generates 128-dimensional encodings.
3. **Distance Matching:** Live encodings are compared against the `encodings.pkl` database using a tolerance-based matching algorithm to ensure precise identification.

## 🚀 Local Setup & Installation
1. **Clone the repository:**
   ```bash
   git clone [https://github.com/mansiggit/Smart-Face-Recognition-System](https://github.com/mansiggit/Smart-Face-Recognition-System)
   cd smart-face-recognition
2. **Install system dependencies (Linux/Ubuntu):**
   ```Bash
   sudo apt-get update && sudo apt-get install cmake build-essential libgl1-mesa-glx
3. **Install Python packages:**
   ```Bash
   pip install -r requirements.txt
4. **Run the App:**
   ```Bash
   streamlit run app.py

## 📈 Performance
* **Latency:** <500ms for identification.
* **Storage:** Minimal footprint using 128-dimensional vector representations.
