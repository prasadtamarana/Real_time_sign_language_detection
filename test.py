import cv2
import numpy as np
from tensorflow.keras.models import load_model

model = load_model("sign_cnn_model.h5")

class_names = np.load("class_names.npy")

IMG_SIZE = 128

image_path = r"C:\Users\prasa\OneDrive\Desktop\Sign_lanug\dataset\C\7.jpg"

image = cv2.imread(image_path)

if image is None:
    print("Image not found")
    exit()

img = cv2.resize(image, (IMG_SIZE, IMG_SIZE))

img = img / 255.0


img = np.expand_dims(img, axis=0)

prediction = model.predict(img)

predicted_class = np.argmax(prediction)
confidence = np.max(prediction)

label = class_names[predicted_class]

print(f"Prediction: {label}")
print(f"Confidence: {confidence * 100:.2f}%")

cv2.putText(
    image,
    f"{label} ({confidence*100:.1f}%)",
    (20, 40),
    cv2.FONT_HERSHEY_SIMPLEX,
    1,
    (0, 255, 0),
    2
)

cv2.imshow("Prediction", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
