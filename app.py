import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import face_recognition
import pickle
import numpy as np
import cv2

# Load your encodings
with open("encodings.pkl", "rb") as f:
    data = pickle.load(f)

class FaceRecognizer(VideoTransformerBase):
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        
        # Recognition Logic (Simplified for Web)
        small_frame = cv2.resize(img, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            matches = face_recognition.compare_faces(data["encodings"], face_encoding, 0.6)
            name = "Unknown"
            if True in matches:
                first_match_index = matches.index(True)
                name = data["names"][first_match_index]

            top *= 4; right *= 4; bottom *= 4; left *= 4
            cv2.rectangle(img, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(img, name, (left, bottom + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        return img

st.title("AI Face Recognition Web Portal")
webrtc_streamer(key="example", video_transformer_factory=FaceRecognizer)