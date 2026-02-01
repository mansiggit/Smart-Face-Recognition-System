import cv2
import os
import sys

def register_person():
    name = input("Enter the name of the person to register: ").strip().replace(" ", "_")
    
    save_path = f"known_faces/{name}"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    # We add cv2.CAP_DSHOW to fix Windows-specific camera lag/loading issues
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    if not cap.isOpened():
        print("❌ Error: Could not open webcam. Try changing the index (e.g., VideoCapture(1))")
        return

    print(f"✅ Webcam opened. Focus the new window and press 's' to save or 'q' to quit.")

    while True:
        ret, frame = cap.read()
        
        if not ret:
            print("❌ Failed to grab frame.")
            break
            
        # Display the preview
        cv2.imshow("Registration - Press 'S' to Save", frame)
        
        # Bring window to front
        cv2.setWindowProperty("Registration - Press 'S' to Save", cv2.WND_PROP_TOPMOST, 1)

        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('s'):
            img_filename = f"{name}_1.jpg"
            full_path = os.path.join(save_path, img_filename)
            cv2.imwrite(full_path, frame)
            print(f"🌟 Success! Image saved at: {full_path}")
            break
        elif key == ord('q'):
            print("Registration cancelled.")
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Ensure we are running from project root
    if not os.path.exists("known_faces"):
        os.makedirs("known_faces")
    register_person()