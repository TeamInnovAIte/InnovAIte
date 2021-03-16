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
import os


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
stop_at = 600
capture_interval = 5  # Average values over 5 second intervals, at 30 fps
session_date = date.today().strftime("%d/%m/%Y")
data_dict = {
    "session_date": session_date,
    "emotion_data": {
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
        "avg_sad_timeline": [],
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
}

def eye_brow_distance(leye,reye):
    global dlib_points
    distq = dist.euclidean(leye,reye)
    dlib_points.append(int(distq))
    return distq

def normalize_values(points,disp):
    try:
        normalized_value = abs(disp - np.min(points))/abs(np.max(points) - np.min(points))
    except ZeroDivisionError:
        return 0
    stress = np.exp(-normalized_value)
    # print(stress_value)
    if math.isnan(stress_value):
        stress = 0
    return stress

while True:
    ret, test_img = cap.read()  # captures frame and returns boolean value and captured image
    if not ret:
        continue
    current_frame += 1
    total_frames += 1

    # test_img = cv2.flip(test_img, 1)
    image_h, image_w, image_l = test_img.shape
    test_img = cv2.resize(test_img, (int(image_w/2), int(image_h/2)))
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

        gray_test_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)
        all_faces = dlib_detector(gray_test_img, 0)
        if len(all_faces) > 0:
            face = all_faces[0]
            # cv2.rectangle(test_img, (x, y), (x + w, y + h), (254, 89, 194), 2)
            # cropped_img = test_img[y - 50:y + h + 50, x - 50:x + w + 50]
            # cv2.imshow('cropped image', cropped_img)

            (lBegin, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eyebrow"]
            (rBegin, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eyebrow"]
            shape = dlib_predictor(test_img, face)
            for i in range(1, 68):  # There are 68 landmark points on each face
                # For each point, draw a red circle with thickness 1 on the original frame
                cv2.circle(test_img, (shape.part(i).x, shape.part(i).y), 1, secondary_color, thickness=2)
            shape = face_utils.shape_to_np(shape)
            leyebrow = shape[lBegin:lEnd]
            reyebrow = shape[rBegin:rEnd]

            reyebrowhull = cv2.convexHull(reyebrow)
            leyebrowhull = cv2.convexHull(leyebrow)

            cv2.drawContours(test_img, [reyebrowhull], -1, secondary_color, 1)
            cv2.drawContours(test_img, [leyebrowhull], -1, secondary_color, 1)

            distq = eye_brow_distance(leyebrow[-1], reyebrow[0])
            stress_value = normalize_values(dlib_points, distq)

        stress_level = ((stress_value + detector.emotions[0]['emotions']['sad'] + detector.emotions[0]['emotions'][
                'angry'] + detector.emotions[0]['emotions']['fear'] + detector.emotions[0]['emotions'][
                      'disgust']) / 6)
        for i, emotion in enumerate(detector.emotions[0]['emotions']):
            cv2.putText(test_img,
                        f"{emotion}: {detector.emotions[0]['emotions'][emotion]}",
                        (detector.emotions[0]['box'][0], detector.emotions[0]['box'][1] - i*15 - 25), font,
                        0.5, main_color, 1)
            cv2.putText(test_img, f"stress: {round(stress_level,2)}", (detector.emotions[0]['box'][0], (detector.emotions[0]['box'][1]+detector.emotions[0]['box'][3]) + 25), font,
                        0.5, main_color, 1)
            data_dict["emotion_data"]["development"][f"{emotion}_timeline"].append(int(detector.emotions[0]['emotions'][emotion] * 100))
        data_dict["emotion_data"]["development"]["stress_timeline"].append(
            int(stress_level * 100)
        )
    else:
        for i, item in enumerate(data_dict["emotion_data"]["development"]):
            data_dict["emotion_data"]["development"][item].append(-1)
        cv2.putText(test_img, f"No detection", (5, 25), font,
                    0.5, main_color, 1)

    # resized_img = cv2.resize(test_img, (1000, 700))
    cv2.imshow('Facial emotion analysis ', test_img)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    if current_frame == stop_at:
        break
cv2.destroyAllWindows()
cap.release()

intervals_total = (current_frame / (frame_rate*capture_interval))

print(f"total frames: {total_frames}")
print(f"total captured frames: {current_frame}")
print(f"total captured intervals: {intervals_total}")
for i, item in enumerate(data_dict["emotion_data"]["development"]):
    print(f"total data points: {len(data_dict['emotion_data']['development'][item])}")

for cur_emotion in ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral", "stress"]:
    total_emotion = 0
    frames_of_no_emotion = 0
    for i in range(len(data_dict["emotion_data"]["development"][f"{cur_emotion}_timeline"])):
        if data_dict["emotion_data"]["development"][f"{cur_emotion}_timeline"][i] != -1:
            total_emotion += data_dict["emotion_data"]["development"][f"{cur_emotion}_timeline"][i]
        else:
            frames_of_no_emotion += 1
        if i % (frame_rate*capture_interval) == 0 and i != 0:
            data_dict["emotion_data"][f"avg_{cur_emotion}_timeline"].append(round(total_emotion/(i-frames_of_no_emotion), 2))
            frames_of_no_emotion = 0
    data_dict["emotion_data"][f"overall_{cur_emotion}"] = round(total_emotion / (len(data_dict["emotion_data"]["development"][f"{cur_emotion}_timeline"])-data_dict["emotion_data"]["development"][f"{cur_emotion}_timeline"].count(-1)), 2)

print(data_dict["emotion_data"]["development"]["stress_timeline"])
if not os.path.exists("emotion_result.json"):
    with open('emotion_result.json', 'w') as json_file:
        json.dump({"sessions": [data_dict]}, json_file)
else:
    with open('emotion_result.json', 'w+') as json_file:
        json_data = json.load(json_file)
        for i, item in enumerate(json_data["sessions"]):
            if json_data["sessions"][i]["session_date"] == session_date:
                json_data["sessions"][i] = data_dict
        json.dump(json_data, json_file)

fig, (ax1, ax2) = plt.subplots(2)
ax1.plot(range(len(data_dict["emotion_data"]["avg_stress_timeline"])), data_dict["emotion_data"]["avg_stress_timeline"])
ax1.set(xlabel=f"Session Duration ({capture_interval} second intervals)", ylabel="Avg. Stress Levels")
ax1.set_title(f"Stress Level Analysis Over Session Time")

ax2.plot(range(len(data_dict["emotion_data"]["development"]["stress_timeline"])), data_dict["emotion_data"]["development"]["stress_timeline"])
ax2.set(xlabel="Session Duration (frames)", ylabel="Stress Levels")
ax2.set_title("Stress Level Analysis Over Session Time (in Frames)")

plt.tight_layout()
plt.show()
