import os
import cv2
import numpy as np
import face_recognition
from tensorflow.keras.models import load_model

# === Paramètres ===
KNOWN_FACES_DIR = "known_faces"
CLIPS_DIR = "media/clips/review"
TOLERANCE = 0.4
SPOOF_MODEL_THRESHOLD = 0.3

# === Chargement modèle anti-spoofing ===
spoof_model = load_model("anti_spoof_model.h5", compile=False)

# === Chargement visages connus ===
print("[INFO] Chargement des visages connus...")
known_encodings = []
known_names = []

for person_name in os.listdir(KNOWN_FACES_DIR):
    person_path = os.path.join(KNOWN_FACES_DIR, person_name)
    for filename in os.listdir(person_path):
        image_path = os.path.join(person_path, filename)
        image = face_recognition.load_image_file(image_path)
        encodings = face_recognition.face_encodings(image)
        if encodings:
            known_encodings.append(encodings[0])
            known_names.append(person_name)

print(f"[INFO] {len(known_encodings)} visages connus chargés.")

# === Traitement des clips Frigate ===
for filename in os.listdir(CLIPS_DIR):
    if not filename.lower().endswith(('.jpg', '.png')):
        continue

    image_path = os.path.join(CLIPS_DIR, filename)
    print(f"\n[INFO] Analyse de {filename}")

    image = cv2.imread(image_path)
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    locations = face_recognition.face_locations(rgb_image)
    encodings = face_recognition.face_encodings(rgb_image, locations)

    for (top, right, bottom, left), encoding in zip(locations, encodings):
        # Reconnaissance
        name = "Inconnu"
        matches = face_recognition.compare_faces(known_encodings, encoding, tolerance=TOLERANCE)
        face_distances = face_recognition.face_distance(known_encodings, encoding)
        if True in matches:
            best_match_index = np.argmin(face_distances)
            name = known_names[best_match_index]

        # Anti-spoofing
        face_image = image[top:bottom, left:right]
        resized_face = cv2.resize(face_image, (224, 224))
        normalized_face = resized_face.astype("float32") / 255.0
        input_face = np.expand_dims(normalized_face, axis=0)
        prediction = spoof_model.predict(input_face)[0][0]

        if prediction < SPOOF_MODEL_THRESHOLD:
            spoof_status = "VRAI visage"
        else:
            spoof_status = "SPOOF détecté"

        print(f"🔍 {name} - {spoof_status} (score: {prediction:.2f})")

print("\n✅ Analyse terminée.")
