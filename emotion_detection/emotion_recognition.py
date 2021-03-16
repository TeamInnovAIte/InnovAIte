from fer import FER
import cv2
import dlib
from scipy.spatial import distance as dist
import math
import json
import numpy as np
from imutils import face_utils
from datetime import datetime, date
import matplotlib.pyplot as plt
import sys

# Prompt for IP camera or regular webcam (This is just for testing because I'm using a RaspberryPi as an IP camera)
input_setting = input("Enter one of the option numbers [1 for IP camera, 2 for Webcam, 3 for Video File]: ")
if input_setting == "1":
    vid_src = "http://jjraspi:9090/?action=stream"
elif input_setting == "2":
    vid_src = 0
else:
    vid_src = input("Enter the video file name to use (example: myvideo.avi): ")
cap = cv2.VideoCapture(vid_src)
font = cv2.FONT_HERSHEY_COMPLEX
detector = FER(mtcnn=True)
dlib_detector = dlib.get_frontal_face_detector()
dlib_predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
dlib_points = []
current_frame = 0
total_frames = 0
frame_rate = 30
main_color = (0, 255, 0)
secondary_color = (255, 7, 58)
capture_interval = 5  # Average values over 5 second intervals, at 30 fps
data_dict = {
    "session_date": date.today().strftime("%d/%m/%Y"),
    "session_time": datetime.now().strftime("%H:%M:%S"),
    "data": {
        "overall_stress": 0,
        "overall_happy": 0,
        "overall_fear": 0,
        "overall_angry": 0,
        "overall_disgust": 0,
        "overall_neutral": 0,
        "overall_surprise": 0,
        "overall_sad": 0,
        "avg_stress_timeline": [],
        "avg_happy_timeline": [],
        "avg_fear_timeline": [],
        "avg_angry_timeline": [],
        "avg_disgust_timeline": [],
        "avg_neutral_timeline": [],
        "avg_surprise_timeline": [],
        "avg_sad_timeline": []
    },
    "development": {
        "stress_timeline": [],
        "happy_timeline": [],
        "fear_timeline": [],
        "angry_timeline": [],
        "disgust_timeline": [],
        "neutral_timeline": [],
        "surprise_timeline": [],
        "sad_timeline": []
    }
}

while True:
    ret, test_img = cap.read()  # captures frame and returns boolean value and captured image
    if not ret:
        continue
    current_frame += 1
    total_frames += 1

    # test_img = cv2.flip(test_img, 1)
    test_img = cv2.resize(test_img, (1000, 700))
    # test_img_dlib = cv2.resize(test_img, (500, 300))
    # test_img = imutils.resize(test_img, width=500, height=500)
    stress_value = 0
    detector.detect_emotions(test_img)
    if len(detector.emotions) > 0:
        cv2.rectangle(test_img,
                      (detector.emotions[0]['box'][0], detector.emotions[0]['box'][1]),
                      (detector.emotions[0]['box'][0] + detector.emotions[0]['box'][2],
                       detector.emotions[0]['box'][1] + detector.emotions[0]['box'][3]),
                      main_color,
                      2)
        stress_level = ((detector.emotions[0]['emotions']['sad'] + detector.emotions[0]['emotions'][
                'angry'] + detector.emotions[0]['emotions']['fear'] + detector.emotions[0]['emotions'][
                      'disgust']) / 5)
        for i, emotion in enumerate(detector.emotions[0]['emotions']):
            cv2.putText(test_img,
                        f"{emotion}: {detector.emotions[0]['emotions'][emotion]}",
                        (detector.emotions[0]['box'][0], detector.emotions[0]['box'][1] - i*15 - 25), font,
                        0.5, main_color, 1)
            cv2.putText(test_img, f"stress: {round(stress_level,2)}", (detector.emotions[0]['box'][0], (detector.emotions[0]['box'][1]+detector.emotions[0]['box'][3]) + 25), font,
                        0.5, main_color, 1)
            data_dict["development"][f"{emotion}_timeline"].append(int(detector.emotions[0]['emotions'][emotion] * 100))
        data_dict["development"]["stress_timeline"].append(
            int(stress_level * 100)
        )
    else:
        for i, item in enumerate(data_dict["development"]):
            data_dict["development"][item].append(-1)
        cv2.putText(test_img, f"No detection", (5, 25), font,
                    0.5, main_color, 1)

    # resized_img = cv2.resize(test_img, (1000, 700))
    cv2.imshow('Facial emotion analysis ', test_img)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
cv2.destroyAllWindows()
cap.release()

intervals_total = (current_frame / (frame_rate*capture_interval))

print(f"total frames: {total_frames}")
print(f"total captured frames: {current_frame}")
print(f"total captured intervals: {intervals_total}")
for i, item in enumerate(data_dict["development"]):
    print(f"total data points: {len(data_dict['development'][item])}")

for cur_emotion in ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral", "stress"]:
    total_emotion = 0
    frames_of_no_emotion = 0
    for i in range(len(data_dict["development"][f"{cur_emotion}_timeline"])):
        if data_dict["development"][f"{cur_emotion}_timeline"][i] != -1:
            total_emotion += data_dict["development"][f"{cur_emotion}_timeline"][i]
        else:
            frames_of_no_emotion += 1
        if i % (frame_rate*capture_interval) == 0 and i != 0:
            data_dict["data"][f"avg_{cur_emotion}_timeline"].append(round(total_emotion/(i-frames_of_no_emotion), 2))
            frames_of_no_emotion = 0
    data_dict["data"][f"overall_{cur_emotion}"] = round(total_emotion / (len(data_dict["development"][f"{cur_emotion}_timeline"])-data_dict["development"][f"{cur_emotion}_timeline"].count(-1)), 2)

print(data_dict["development"]["stress_timeline"])
with open('emotion_result.json', 'w') as json_file:
    json.dump(data_dict, json_file)

fig, (ax1, ax2) = plt.subplots(2)
ax1.plot(range(len(data_dict["data"]["avg_stress_timeline"])), data_dict["data"]["avg_stress_timeline"])
ax1.set(xlabel=f"Session Duration ({capture_interval} second intervals)", ylabel="Avg. Stress Levels")
ax1.set_title(f"Stress Level Analysis Over Session Time")

ax2.plot(range(len(data_dict["development"]["stress_timeline"])), data_dict["development"]["stress_timeline"])
ax2.set(xlabel="Session Duration (frames)", ylabel="Stress Levels")
ax2.set_title("Stress Level Analysis Over Session Time (in Frames)")

plt.tight_layout()
plt.show()
