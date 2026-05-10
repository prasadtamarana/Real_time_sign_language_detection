import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model
from collections import deque

model = load_model("sign_cnn_model.h5")
class_names = np.load("class_names.npy")

IMG_SIZE = 128
CONFIDENCE_THRESHOLD = 0.75

STABLE_FRAMES = 8
prediction_queue = deque(maxlen=STABLE_FRAMES)

final_text = ""
last_added_letter = ""

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

with mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as hands:

    while True:
        ret, frame = cap.read()

        if not ret:
            print("Camera not working")
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        label = "No Hand"
        skeleton_img = np.ones((400, 400, 3), dtype=np.uint8) * 255

        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]

            mp_draw.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_draw.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                mp_draw.DrawingSpec(color=(0, 255, 0), thickness=2)
            )

            mp_draw.draw_landmarks(
                skeleton_img,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_draw.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                mp_draw.DrawingSpec(color=(0, 255, 0), thickness=2)
            )

            img = cv2.resize(skeleton_img, (IMG_SIZE, IMG_SIZE))
            img = img.astype("float32") / 255.0
            img = np.expand_dims(img, axis=0)

            prediction = model.predict(img, verbose=0)[0]

            predicted_class = np.argmax(prediction)
            confidence = prediction[predicted_class]
            predicted_label = class_names[predicted_class]

            if confidence >= CONFIDENCE_THRESHOLD:
                label = predicted_label
                prediction_queue.append(predicted_label)

                if len(prediction_queue) == STABLE_FRAMES:
                    most_common = max(set(prediction_queue), key=prediction_queue.count)

                    if prediction_queue.count(most_common) >= STABLE_FRAMES - 2:
                        if most_common != last_added_letter:
                            final_text += most_common
                            last_added_letter = most_common
                            prediction_queue.clear()
            else:
                label = f"Unknown / Guess: {predicted_label}"
                prediction_queue.clear()

            cv2.putText(
                frame,
                f"Letter: {label} ({confidence*100:.1f}%)",
                (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                3
            )

        else:
            prediction_queue.clear()
            last_added_letter = ""

        cv2.rectangle(frame, (10, 420), (620, 470), (255, 255, 255), -1)
        cv2.putText(
            frame,
            f"Text: {final_text}",
            (20, 455),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 0),
            2
        )

        cv2.putText(
            frame,
            "Controls: q=Quit | c=Clear | b=Backspace | Space=Add Space",
            (20, 500),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )

        cv2.imshow("Webcam", frame)
        cv2.imshow("Skeleton Input To CNN", skeleton_img)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break
        elif key == ord("c"):
            final_text = ""
            last_added_letter = ""
            prediction_queue.clear()
        elif key == ord("b"):
            final_text = final_text[:-1]
            last_added_letter = ""
            prediction_queue.clear()
        elif key == ord(" "):
            final_text += " "
            last_added_letter = ""
            prediction_queue.clear()

cap.release()
cv2.destroyAllWindows()