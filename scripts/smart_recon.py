import cv2
import face_recognition
import pickle
import numpy as np
import pyttsx3
import threading
from datetime import datetime
import time

# Initialize Text-to-Speech Engine
engine = pyttsx3.init()
engine.setProperty('rate', 150) # Speed of speech

def speak(text):
    """Function to speak without freezing the video feed"""
    def run_speech():
        engine.say(text)
        engine.runAndWait()
    threading.Thread(target=run_speech).start()

def main():
    # Load Data
    with open("encodings.pkl", "rb") as f:
        data = pickle.load(f)
    known_encodings, known_names = data["encodings"], data["names"]

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    
    # Motion Detection Variables
    prev_frame = None
    motion_threshold = 10000  # Sensitivity: Lower = more sensitive
    
    last_logged = {}
    log_cooldown = 60 # Seconds

    print("🤖 Smart AI System Online. Waiting for motion...")

    while True:
        ret, frame = cap.read()
        if not ret: break

        # --- STEP 1: MOTION DETECTION ---
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)

        if prev_frame is None:
            prev_frame = gray
            continue

        # Calculate difference between current and previous frame
        frame_delta = cv2.absdiff(prev_frame, gray)
        thresh = cv2.threshold(frame_delta, 25, 255, cv2.THRESH_BINARY)[1]
        motion_value = np.sum(thresh)
        prev_frame = gray

        # Only recognize faces if motion is detected
        if motion_value > motion_threshold:
            cv2.putText(frame, "STATUS: MOTION DETECTED", (10, 20), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            # --- STEP 2: FACE RECOGNITION ---
            small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
            rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
            
            face_locations = face_recognition.face_locations(rgb_small_frame)
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

            for face_encoding, (top, right, bottom, left) in zip(face_encodings, face_locations):
                matches = face_recognition.compare_faces(known_encodings, face_encoding, 0.5)
                name = "Unknown"

                face_distances = face_recognition.face_distance(known_encodings, face_encoding)
                if len(face_distances) > 0:
                    best_match_index = np.argmin(face_distances)
                    if matches[best_match_index]:
                        name = known_names[best_match_index].replace("_", " ")

                        # --- STEP 3: VOICE & LOGGING ---
                        curr_time = time.time()
                        if name not in last_logged or (curr_time - last_logged[name] > log_cooldown):
                            speak(f"Hello {name}, welcome back.")
                            last_logged[name] = curr_time

                # Drawing (Corners)
                top *= 4; right *= 4; bottom *= 4; left *= 4
                cv2.rectangle(frame, (left, top), (right, bottom), (255, 255, 0), 2)
                cv2.putText(frame, name, (left, bottom + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        else:
            cv2.putText(frame, "STATUS: IDLE (SAVING POWER)", (10, 20), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        cv2.imshow('Smart Face AI', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()