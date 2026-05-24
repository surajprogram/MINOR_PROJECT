import cv2
import face_recognition
import pickle
import os
import threading
import time


def open_camera():
    backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]
    for backend in backends:
        for index in range(2):
            cap = cv2.VideoCapture(index, backend)
            if cap.isOpened():
                print(f"Opened camera index {index} with backend {backend}")
                return cap
            cap.release()
    return None


def check_camera():
    cap = open_camera()
    if cap is None:
        return False, "Failed to open webcam. Check camera connection, permissions, and whether another app is using the camera."
    cap.release()
    return True, "Webcam opened successfully."


def recognize_faces(timeout=10):
    """
    Recognize faces from webcam with timeout
    timeout: Maximum time to search for face (in seconds)
    """
    # Get the directory of this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    encodings_path = os.path.join(script_dir, "../data/encodings.pkl")
    
    if not os.path.exists(encodings_path):
        return None
    
    try:
        with open(encodings_path, "rb") as f:
            data = pickle.load(f)
    except Exception as e:
        print(f"Error loading encodings: {e}")
        return None

    video = open_camera()
    if video is None:
        print("Failed to open webcam. Check camera connection, permissions, and whether another app is using the camera.")
        return None

    start_time = time.time()
    recognized_name = None

    tolerance = 0.48

    while True:
        elapsed_time = time.time() - start_time
        if elapsed_time > timeout:
            print("Camera timeout reached without detecting a face.")
            break
        
        ret, frame = video.read()
        if not ret or frame is None:
            print("Failed to grab a frame from the webcam.")
            break
        
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        faces = face_recognition.face_locations(rgb)

        if len(faces) != 1:
            continue

        encodings = face_recognition.face_encodings(rgb, faces)

        for encoding in encodings:
            distances = face_recognition.face_distance(data["encodings"], encoding)
            matches = list(distances <= tolerance)

            if not any(matches):
                continue

            matched_idxs = [i for i, is_match in enumerate(matches) if is_match]
            best_idx = min(matched_idxs, key=lambda i: distances[i])
            recognized_name = data["names"][best_idx]
            print(f"Recognized {recognized_name} with distance {distances[best_idx]:.4f}")
            break

        if recognized_name:
            break

    video.release()
    cv2.destroyAllWindows()
    return recognized_name