import face_recognition
import cv2
import pickle

# Load trained data
with open("data/encodings.pkl", "rb") as f:
    encodings, names = pickle.load(f)

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    rgb = frame[:, :, ::-1]

    faces = face_recognition.face_locations(rgb)
    face_encodings = face_recognition.face_encodings(rgb, faces)

    for encoding in face_encodings:
        matches = face_recognition.compare_faces(encodings, encoding)

        name = "Unknown"

        if True in matches:
            index = matches.index(True)
            name = names[index]

        print("Detected:", name)

    cv2.imshow("Camera", frame)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()