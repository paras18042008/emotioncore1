# EmotionCore v0.3 - Intelligent Desktop App with Face Mesh + Accurate Mood Detection
# Author: OpenAI x Paras

import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from collections import deque

# ----- CONFIG -----
MOOD_PALETTE = ["Happy", "Calm", "Tired", "Stressed", "Sad", "Focused"]
MOOD_THRESHOLD = 10  # Frames required to confirm a new mood
FONT = cv2.FONT_HERSHEY_SIMPLEX

# ----- Mediapipe Setup -----
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# ----- Load Emotion Model -----
model = tf.keras.models.load_model("fer_model.h5", compile=False)
labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# ----- Helpers -----
def preprocess_face(face_img):
    face = cv2.resize(face_img, (64, 64))
    gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    gray = gray.astype("float") / 255.0
    gray = np.expand_dims(gray, axis=-1)
    return np.expand_dims(gray, axis=0)

def map_emotion_to_mood(emotion):
    mapping = {
        "Happy": "Happy",
        "Sad": "Sad",
        "Neutral": "Calm",
        "Angry": "Stressed",
        "Disgust": "Stressed",
        "Fear": "Tired",
        "Surprise": "Focused"
    }
    return mapping.get(emotion, "Calm")

# ----- Mood Smoothing Buffer -----
frame_buffer = deque(maxlen=MOOD_THRESHOLD)
last_confirmed_mood = "Calm"

# ----- Webcam Setup -----
cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    # Detect face region for emotion prediction
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    mood = last_confirmed_mood
    for (x, y, w, h) in faces:
        face_img = frame[y:y+h, x:x+w]
        if face_img.size == 0:
            continue
        input_face = preprocess_face(face_img)
        preds = model.predict(input_face, verbose=0)
        emotion = labels[np.argmax(preds)]
        mood = map_emotion_to_mood(emotion)
        frame_buffer.append(mood)
        break  # only use first detected face for mood

    # Confirm mood over multiple frames
    if len(frame_buffer) == MOOD_THRESHOLD:
        most_common = max(set(frame_buffer), key=frame_buffer.count)
        if frame_buffer.count(most_common) > MOOD_THRESHOLD // 2:
            last_confirmed_mood = most_common

    # Draw face mesh and mood
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            mp_drawing.draw_landmarks(
                image=frame,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style())

    # Draw mood
    mood_text = f"Mood: {last_confirmed_mood}"
    text_size = cv2.getTextSize(mood_text, FONT, 1.2, 2)[0]
    text_x = int((frame.shape[1] - text_size[0]) / 2)
    text_y = frame.shape[0] - 30
    cv2.rectangle(frame, (text_x - 10, text_y - 40), (text_x + text_size[0] + 10, text_y + 10), (0, 0, 0), -1)
    cv2.putText(frame, mood_text, (text_x, text_y), FONT, 1.2, (255, 255, 255), 2)

    cv2.imshow('EmotionCore v0.3 - Fast + Accurate', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
