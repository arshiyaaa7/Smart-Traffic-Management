import numpy as np
import tensorflow as tf
import cv2
from tensorflow.keras.utils import get_custom_objects

frameWidth = 640
frameHeight = 480
cap = cv2.VideoCapture('ambulance.mp4')
# cap = cv2.VideoCapture(1)
cap.set(3, frameWidth)
cap.set(4, frameHeight)
cap.set(10, 150)

data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
np.set_printoptions(suppress=True)


model = tf.keras.models.load_model('keras_model.h5', custom_objects=custom_objects)


def emergency():
    # Removed audio processing and FFT part
    pass

amc = 0
while True:
    success, img = cap.read()
    if not success:
        break

    img = cv2.resize(img, (224, 224))
    img_res = np.asarray(img)
    normalized_image_array = (img_res.astype(np.float32) / 127.0) - 1
    data[0] = normalized_image_array
    prediction = model.predict(data)

    if prediction.shape[1] >= 4:  # Ensure the prediction shape is correct
        ambulance_acc = prediction[0][0]
        firengin_acc = []

        ambulance = f'ambulance:{prediction[0][0].round()}'
        firengine = f'firengine:{prediction[0][1].round()}'
        trafic = f'trafic:{prediction[0][3].round()}'

        if ambulance == 'ambulance:1.0':
            amc += 1
        if amc == 20:
            amc = 0
            emergency()

        if firengine == 'firengine:1.0':
            cv2.putText(img, 'Firengine detected', (450, 100), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)

        text = f'{ambulance}% {firengine}% {trafic}% {amc}'
        cv2.putText(img, text, (15, 20), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 2)
        cv2.imshow("Result", img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        print("Model prediction output shape is not as expected.")

cap.release()
cv2.destroyAllWindows()
