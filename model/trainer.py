import face_recognition
import os
import pickle

def train_model():
    path = "data/images"
    encodings = []
    names = []

    for person in os.listdir(path):
        person_path = os.path.join(path, person)

        for img_name in os.listdir(person_path):
            img_path = os.path.join(person_path, img_name)

            image = face_recognition.load_image_file(img_path)
            face_enc = face_recognition.face_encodings(image)

            if face_enc:
                encodings.append(face_enc[0])
                names.append(person)

    with open("data/encodings.pkl", "wb") as f:
        pickle.dump((encodings, names), f)

    print("✅ Training Done")

# RUN THIS
train_model()