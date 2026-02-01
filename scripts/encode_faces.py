import face_recognition
import pickle
import os
import cv2

def generate_encodings():
    known_encodings = []
    known_names = []
    base_path = "known_faces"

    print("🚀 Starting encoding process...")

    # Loop through every person's folder in 'known_faces'
    for person_name in os.listdir(base_path):
        person_dir = os.path.join(base_path, person_name)
        
        if not os.path.isdir(person_dir):
            continue

        print(f"📸 Processing: {person_name}")

        # Loop through every image in that person's folder
        for img_name in os.listdir(person_dir):
            img_path = os.path.join(person_dir, img_name)
            
            # Load the image
            image = face_recognition.load_image_file(img_path)
            
            # Find face locations and encodings
            # We use 'hog' for CPU, but 'cnn' is more accurate if you have a GPU
            face_locations = face_recognition.face_locations(image, model="hog")
            face_encodings = face_recognition.face_encodings(image, face_locations)

            if len(face_encodings) > 0:
                known_encodings.append(face_encodings[0])
                known_names.append(person_name)
                print(f"   ✅ Encoded {img_name}")
            else:
                print(f"   ⚠️ No face found in {img_name}. Skipping.")

    # Save the data to a file
    data = {"encodings": known_encodings, "names": known_names}
    with open("encodings.pkl", "wb") as f:
        f.write(pickle.dumps(data))
    
    print("\n✨ Success! 'encodings.pkl' has been created.")
    print(f"Total faces encoded: {len(known_names)}")

if __name__ == "__main__":
    generate_encodings()