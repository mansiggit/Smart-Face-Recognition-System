import cv2
import face_recognition
import pickle
import numpy as np
import time
import csv
from datetime import datetime
import os

# --- NEW: Logging Function ---
def log_attendance(name):
    file_path = "attendance_log.csv"
    
    # Check if file exists to write header
    file_exists = os.path.isfile(file_path)
    
    with open(file_path, mode='a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(['Name', 'Date', 'Time']) # Header
        
        now = datetime.now()
        date_str = now.strftime("%Y-%m-%d")
        time_str = now.strftime("%H:%M:%S")
        writer.writerow([name, date_str, time_str])
        print(f"📝 Logged: {name} at {time_str}")

def main():
    # Load Encodings
    try:
        with open("encodings.pkl", "rb") as f:
            data = pickle.load(f)
    except FileNotFoundError:
        print("❌ Encodings not found!")
        return

    known_encodings = data["encodings"]
    known_names = data["names"]

    # --- NEW: Cooldown Logic ---
    # Dictionary to keep track of when a person was last logged
    # { "Mansi": timestamp }
    last_logged = {} 
    log_cooldown = 30 # Seconds to wait before logging the same person again

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    process_this_frame = True
    prev_frame_time = 0

    while True:
        ret, frame = cap.read()
        if not ret: break

        if process_this_frame:
            small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
            rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
            face_locations = face_recognition.face_locations(rgb_small_frame)
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

            face_names = []
            for face_encoding in face_encodings:
                matches = face_recognition.compare_faces(known_encodings, face_encoding, tolerance=0.5)
                name = "Unknown"

                face_distances = face_recognition.face_distance(known_encodings, face_encoding)
                if len(face_distances) > 0:
                    best_match_index = np.argmin(face_distances)
                    if matches[best_match_index]:
                        name = known_names[best_match_index].replace("_", " ")
                        
                        # --- NEW: Check Cooldown and Log ---
                        current_time = time.time()
                        if name not in last_logged or (current_time - last_logged[name] > log_cooldown):
                            log_attendance(name)
                            last_logged[name] = current_time
                
                face_names.append(name)

        process_this_frame = not process_this_frame

        # (Drawing Logic - Same as before)
        for (top, right, bottom, left), name in zip(face_locations, face_names):
            top *= 4; right *= 4; bottom *= 4; left *= 4
            color = (255, 255, 0) if name != "Unknown" else (0, 0, 255)
            # Corners
            cv2.line(frame, (left, top), (left+30, top), color, 3)
            cv2.line(frame, (left, top), (left, top+30), color, 3)
            cv2.line(frame, (right, top), (right-30, top), color, 3)
            cv2.line(frame, (right, top), (right, top+30), color, 3)
            # Name
            cv2.putText(frame, name, (left, bottom + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        cv2.imshow('Face Recognition & Logger', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()