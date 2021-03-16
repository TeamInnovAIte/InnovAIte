import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
import cv2
from datetime import datetime, date
import matplotlib.pyplot as plt
import json


def get_model():
    # Create a simple model.
    inputs = keras.Input(shape=(32,))
    outputs = keras.layers.Dense(1)(inputs)
    model = keras.Model(inputs, outputs)
    model.compile(optimizer="adam", loss="mean_squared_error")
    return model


model = keras.models.Sequential()
model.add(keras.layers.InputLayer(
    input_shape=(128, 128, 3)
))

model.add(
    keras.layers.Conv2D(
        filters=32,
        kernel_size=(5, 5),
        strides=(1, 1),
        padding='same',
        activation='relu',
        name='Conv_1'))

model.add(
    keras.layers.MaxPool2D(
        pool_size=(2, 2),
        name='Pool_1'))  # Image_size: 32*64*64(32 filters,image_size 64*64)

model.add(
    keras.layers.Conv2D(
        filters=64,
        kernel_size=(5, 5),
        strides=(1, 1),
        padding='same',
        activation='relu',
        name='Conv_2'))

model.add(
    keras.layers.MaxPool2D(
        pool_size=(2, 2),
        name='Pool_2'))  # Image_size: 64*32*32(64 filters,image_size 32*32)

model.add(
    keras.layers.Conv2D(
        filters=128,
        kernel_size=(5, 5),
        strides=(1, 1),
        padding='same',
        activation='relu',
        name='Conv_3'))

model.add(
    keras.layers.MaxPool2D(
        pool_size=(2, 2),
        name='Pool_3'))  # Image_size: 128*16*16(128 filters,image_size 16*16)

model.add(
    keras.layers.Conv2D(
        filters=256,
        kernel_size=(5, 5),
        strides=(1, 1),
        padding='same',
        activation='relu',
        name='Conv_4'))

model.add(
    keras.layers.MaxPool2D(
        pool_size=(2, 2),
        name='Pool_4'))  # Image_size: 256*8*8(256 filters,image_size 8*8)

model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(units=1024, activation='relu', name='fc_1'))
model.add(keras.layers.Dropout(rate=0.2))
model.add(keras.layers.Dense(units=512, activation='relu', name='fc_2'))
model.add(keras.layers.Dense(units=10, activation='softmax', name='fc_3'))
model.save('/tmp/model')

model.built = True
model.load_weights("Train_weights_1.h5")

# function that takes path to the image- and above lines
# c0- safe driving
# c1- texting
# c2- talking on phone
# c3- operating center console
# c4- drinking
# c5- reaching behind
# c6- hair/makeup
# c7- talking to passenger
classes = ["safe_driving", "texting", "talking_on_phone", "operating_center_console", "drinking", "reaching_behind",
           "hair_makeup", "talking_to_passenger"]
data_dict = {
    "session_date": date.today().strftime("%d/%m/%Y"),
    "session_time": datetime.now().strftime("%H:%M:%S"),
    "data": {

    },
    "development": {
        "safe_driving": [],
        "texting": [],
        "talking_on_phone": [],
        "operating_center_console": [],
        "drinking": [],
        "reaching_behind": [],
        "hair_makeup": [],
        "talking_to_passenger": []
    }
}

def output_label(predict):
    if predict == 1 or predict == 3:
        predict = 1
    elif predict == 2 or predict == 4:
        predict = 2
    else:
        predict -= 2
    if predict < len(classes):
        return classes[predict]
    return "N/A"


# Prompt for IP camera or regular webcam (This is just for testing because I'm using a RaspberryPi as an IP camera)
input_setting = input("Enter one of the option numbers [1 for IP camera, 2 for Webcam, 3 for Video File]: ")
if input_setting == "1":
    vid_src = "http://jjraspi:9090/?action=stream"
elif input_setting == "2":
    vid_src = 0
else:
    vid_src = input("Enter the video file name to use (example: myvideo.avi): ")
# Start capturing the video feed
cap = cv2.VideoCapture(vid_src)
current_frame = 0
frame_rate = 30
capture_interval = 10  # Average values over 10 second intervals, at 30 fps

# Continuously capture images from the webcam and analyze the images.
while True:
    ret, test_img = cap.read()  # captures frame and returns boolean value and captured image
    if not ret:
        continue
    # Keeps track of the number of frames captured.
    current_frame += 1

    # test_img = cv2.flip(test_img, 1)
    test_img_resized = cv2.resize(test_img, (128, 128))
    # Make a gesture prediction and display it to the screen.
    test_img_predict = np.reshape(test_img_resized,
                                  [1, 128, 128, 3])  # 128 by 128 dimension, 3 because 3 channel rgb for color
    img_arr_predict = model.predict(test_img_predict, batch_size=1)
    label = output_label(np.argmax(img_arr_predict))
    cv2.putText(test_img,
                f"Detected Gesture: {label}", (0, 25), cv2.FONT_HERSHEY_SIMPLEX, 1,
                (254, 89, 194), 2)
    # Add the detected gesture into the timeline.
    for gesture_type in classes:
        if gesture_type == label:
            data_dict["development"][f"{label}"].append(1)
        else:
            data_dict["development"][f"{gesture_type}"].append(0)

    resized_img = cv2.resize(test_img, (1000, 700))
    cv2.imshow('Gesture Analysis', resized_img)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
cv2.destroyAllWindows()
cap.release()

intervals_total = (current_frame / (frame_rate*capture_interval))

print(f"total captured frames: {current_frame}")
print(f"total captured intervals: {intervals_total}")
for i, item in enumerate(data_dict["development"]):
    print(f"{item} total data points: {len(data_dict['development'][item])}")

for gesture_type in classes:
    print(f"""{gesture_type} detections: {data_dict['development'][f'{gesture_type}'].count(1)}""")

with open('gesture_result.json', 'w') as json_file:
    json.dump(data_dict, json_file)


class_counts = [i for i, _ in enumerate(classes)]
gesture_counts = []
for gesture_type in classes:
    gesture_counts.append(data_dict['development'][f'{gesture_type}'].count(1))

plt.style.use('ggplot')
plt.bar(class_counts, gesture_counts, color='blue')
plt.xlabel(f"Gesture types")
plt.ylabel("Number of gesture detections")
plt.title(f"Gesture Analysis Over Session Time")
plt.xticks(class_counts, classes)

plt.show()
