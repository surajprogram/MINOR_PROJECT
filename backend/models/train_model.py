import face_recognition
import os
import pickle
from PIL import Image

# Get the directory of this script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(SCRIPT_DIR, "../data/data/images")
ENCODINGS_PATH = os.path.join(SCRIPT_DIR, "../data/encodings.pkl")

known_encodings = []
known_names = []
encoding_sources = []

print(f"📁 Loading images from: {DATASET_PATH}")
print("-" * 50)

# Check if dataset path exists
if not os.path.exists(DATASET_PATH):
    print(f"❌ Error: Dataset folder not found at {DATASET_PATH}")
    exit(1)

# Get list of person folders
person_folders = [f for f in os.listdir(DATASET_PATH) if os.path.isdir(os.path.join(DATASET_PATH, f))]

if not person_folders:
    print("❌ No person folders found in dataset!")
    exit(1)

print(f"✅ Found {len(person_folders)} person folders\n")


def save_encodings():
    data = {
        "encodings": known_encodings,
        "names": known_names,
    }
    with open(ENCODINGS_PATH, "wb") as f:
        pickle.dump(data, f)
    print(f"💾 Encodings saved to: {ENCODINGS_PATH}")


def predict_name(test_encoding, tolerance=0.6, exclude_index=None):
    distances = face_recognition.face_distance(known_encodings, test_encoding)
    candidates = [
        (idx, dist)
        for idx, dist in enumerate(distances)
        if idx != exclude_index
    ] if exclude_index is not None else list(enumerate(distances))

    if not candidates:
        return "Unknown"

    best_idx, best_dist = min(candidates, key=lambda item: item[1])
    return known_names[best_idx] if best_dist <= tolerance else "Unknown"


def evaluate_training_accuracy():
    if not encoding_sources:
        print("⚠️  No encodings available for accuracy evaluation.")
        return

    total = 0
    correct = 0
    invalid = 0

    for idx, (true_name, img_path) in enumerate(encoding_sources):
        image = face_recognition.load_image_file(img_path)
        encodings = face_recognition.face_encodings(image, num_jitters=1)

        if not encodings:
            invalid += 1
            continue

        predicted_name = predict_name(encodings[0], exclude_index=idx)
        total += 1

        if predicted_name == true_name:
            correct += 1

    if total == 0:
        print("⚠️  No valid faces were evaluated for accuracy.")
        return

    accuracy = (correct / total) * 100
    print("-" * 50)
    print(f"🎯 Training evaluation results:")
    print(f"  Total valid images evaluated: {total}")
    print(f"  Correct predictions: {correct}")
    print(f"  Invalid/undetected faces: {invalid}")
    print(f"  Accuracy: {accuracy:.2f}%")


try:
    # Process each person folder
    for person_name in person_folders:
        person_folder = os.path.join(DATASET_PATH, person_name)
        image_files = [f for f in os.listdir(person_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.gif', '.bmp'))]

        if not image_files:
            print(f"⚠️  {person_name}: No images found")
            continue

        print(f"📸 Processing {person_name} ({len(image_files)} images)...")
        person_encodings_count = 0

        for idx, img_name in enumerate(image_files, start=1):
            try:
                img_path = os.path.join(person_folder, img_name)
                image = face_recognition.load_image_file(img_path)

                # Use a single jitter for speed and reliable results
                encodings = face_recognition.face_encodings(image, num_jitters=1)

                if len(encodings) > 0:
                    known_encodings.append(encodings[0])
                    known_names.append(person_name)
                    encoding_sources.append((person_name, img_path))
                    person_encodings_count += 1
                    print(f"  [{idx}/{len(image_files)}] ✅ {img_name}: face encoded")
                else:
                    print(f"  [{idx}/{len(image_files)}] ⚠️  {img_name}: No face detected")

            except KeyboardInterrupt:
                print("\n⏸️  Training interrupted by user.")
                raise
            except Exception as e:
                print(f"  [{idx}/{len(image_files)}] ❌ {img_name}: Error - {str(e)}")
                continue

        if person_encodings_count > 0:
            print(f"  ✅ {person_name}: {person_encodings_count} faces encoded\n")
        else:
            print(f"  ❌ {person_name}: No valid faces found\n")

    print("-" * 50)
    print(f"\n✅ Total faces encoded: {len(known_encodings)}")

    if len(known_encodings) == 0:
        print("❌ No faces found! Add images and try again.")
        exit(1)

    save_encodings()
    evaluate_training_accuracy()
    print("✅ Training completed successfully!")

except KeyboardInterrupt:
    if known_encodings:
        print("\n💾 Saving partially computed encodings before exit...")
        save_encodings()
        print("⚠️  Partial encodings saved. Re-run training to add more data.")
    else:
        print("⚠️  No encodings were computed. No file saved.")
    exit(1)
