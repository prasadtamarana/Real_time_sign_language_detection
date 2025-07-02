import cv2
import os

DATA_DIR = r'C:\Users\prasa\Downloads\asl_alphabet_test\asl_alphabet_test'
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

labels = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
cap = cv2.VideoCapture(0)

for label in labels:
    folder_path = os.path.join(DATA_DIR, label)
    os.makedirs(folder_path, exist_ok=True)
    print(f'Collecting images for: {label}')
    count = 0
    while count < 300:  # 300 samples per letter
        ret, frame = cap.read()
        if not ret:
            continue
        frame = cv2.flip(frame, 1)
        cv2.rectangle(frame, (100,100), (300,300), (0,255,0), 2)
        roi = frame[100:300, 100:300]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        img_path = os.path.join(folder_path, f"{count}.jpg")
        cv2.imwrite(img_path, gray)
        count += 1
        cv2.imshow("Dataset Collection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()


import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator

img_size = 64
batch_size = 32

# Data generators
train_gen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train_data = train_gen.flow_from_directory(
    r"C:\Users\prasa\Downloads\asl_alphabet_test\asl_alphabet_test",
    target_size=(img_size, img_size),
    color_mode='grayscale',
    batch_size=batch_size,
    class_mode='categorical',
    subset='training'
)

val_data = train_gen.flow_from_directory(
    r"C:\Users\prasa\Downloads\asl_alphabet_test\asl_alphabet_test",
    target_size=(img_size, img_size),
    color_mode='grayscale',
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation'
)

# CNN model
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(img_size, img_size, 1)),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(26, activation='softmax')  # A–Z
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(train_data, validation_data=val_data, epochs=10)

model.save('asl_model.h5')


import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model

model = load_model("asl_model.h5")
labels = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False,
                       max_num_hands=1,
                       min_detection_confidence=0.5)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    h, w, c = frame.shape
    framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(framergb)

    if result.multi_hand_landmarks:
        for handlm in result.multi_hand_landmarks:
            x_min = w
            y_min = h
            x_max = y_max = 0
            for lm in handlm.landmark:
                x, y = int(lm.x * w), int(lm.y * h)
                x_min = min(x_min, x)
                y_min = min(y_min, y)
                x_max = max(x_max, x)
                y_max = max(y_max, y)

            x_min = max(0, x_min - 20)
            y_min = max(0, y_min - 20)
            x_max = min(w, x_max + 20)
            y_max = min(h, y_max + 20)

            roi = frame[y_min:y_max, x_min:x_max]
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            resized = cv2.resize(gray, (64, 64))
            norm = resized / 255.0
            reshaped = np.reshape(norm, (1, 64, 64, 1))

            pred = model.predict(reshaped)
            class_index = np.argmax(pred)
            predicted_letter = labels[class_index]

            cv2.putText(frame, predicted_letter, (x_min, y_min-10), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,255,0), 2)
            mp_draw.draw_landmarks(frame, handlm, mp_hands.HAND_CONNECTIONS)

    cv2.imshow("Sign Language Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


import pyttsx3

engine = pyttsx3.init()
voices = engine.getProperty('voices')
for voice in voices:
    print(f"Voice: {voice.name}, ID: {voice.id}")
