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

useIPCamera = input("Use IP camera? [Y/N]")
if useIPCamera == "Y":
    cam_url = "http://jjraspi:9090/?action=stream"
else:
    cam_url = 0
cap = cv2.VideoCapture(cam_url)
detector = FER(mtcnn=True)
dlib_detector = dlib.get_frontal_face_detector()
dlib_predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
dlib_points = []
current_frame = 0
total_frames = 0
frame_rate = 30
capture_interval = 10  # Average values over 10 second intervals, at 30 fps
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

def eye_brow_distance(leye,reye):
    global dlib_points
    distq = dist.euclidean(leye,reye)
    dlib_points.append(int(distq))
    return distq

def normalize_values(points,disp):
    normalized_value = abs(disp - np.min(points))/abs(np.max(points) - np.min(points))
    stress_value = np.exp(-(normalized_value))
    # print(stress_value)
    if math.isnan(stress_value):
        stress_value = 0
    if stress_value>=75:
        return stress_value,"High Stress"
    else:
        return stress_value,"low_stress"

while True:
    ret, test_img = cap.read()# captures frame and returns boolean value and captured image
    if not ret:
        continue
    current_frame += 1
    total_frames += 1

    test_img = cv2.flip(test_img, 1)
    test_img = cv2.resize(test_img, (1000, 700))
    # test_img = imutils.resize(test_img, width=500, height=500)
    (lBegin, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eyebrow"]
    (rBegin, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eyebrow"]
    (lLower, rUpper) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]
    gray_test_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)

    dlib_detections = dlib_detector(gray_test_img, 0)
    detector.detect_emotions(test_img)
    # print(detector.emotions)

    if len(dlib_detections) == 0:
        continue

    stress_value = 0
    for detection in dlib_detections:
        shape = dlib_predictor(test_img, detection)
        for i in range(1, 68):  # There are 68 landmark points on each face
            # For each point, draw a red circle with thickness 1 on the original frame
            cv2.circle(test_img, (shape.part(i).x, shape.part(i).y), 1, (0, 255, 0), thickness=2)

        shape = face_utils.shape_to_np(shape)
        leyebrow = shape[lBegin:lEnd]
        reyebrow = shape[rBegin:rEnd]

        reyebrowhull = cv2.convexHull(reyebrow)
        leyebrowhull = cv2.convexHull(leyebrow)

        cv2.drawContours(test_img, [reyebrowhull], -1, (0, 255, 0), 1)
        cv2.drawContours(test_img, [leyebrowhull], -1, (0, 255, 0), 1)

        distq = eye_brow_distance(leyebrow[-1], reyebrow[0])
        stress_value, stress_label = normalize_values(dlib_points, distq)

    if len(detector.emotions) > 0:
        image = cv2.rectangle(test_img,
                              (detector.emotions[0]['box'][0], detector.emotions[0]['box'][1]),
                              (detector.emotions[0]['box'][0] + detector.emotions[0]['box'][2],
                               detector.emotions[0]['box'][1] + detector.emotions[0]['box'][3]),
                              (254, 89, 194),
                              2)
        font = cv2.FONT_HERSHEY_SIMPLEX

        for i, emotion in enumerate(detector.emotions[0]['emotions']):
            cv2.putText(test_img,
                        f"{emotion}: {detector.emotions[0]['emotions'][emotion]}",
                        (detector.emotions[0]['box'][0], detector.emotions[0]['box'][1] - i*15 - 25), font, 0.5, (254, 89, 194), 1)
            data_dict["development"][f"{emotion}_timeline"].append(int(detector.emotions[0]['emotions'][emotion] * 100))
        data_dict["development"]["stress_timeline"].append(int(stress_value * 100))

    resized_img = cv2.resize(test_img, (1000, 700))
    cv2.imshow('Facial emotion analysis ', resized_img)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
cv2.destroyAllWindows()
cap.release()

minutes_total = (current_frame / (frame_rate*capture_interval))

print(f"total frames: {total_frames}")
print(f"total captured frames: {current_frame}")
print(f"total captured minutes: {minutes_total}")
for i, item in enumerate(data_dict["development"]):
    print(f"total data points: {len(data_dict['development'][item])}")

for cur_emotion in ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral", "stress"]:
    total_emotion = 0
    for i in range(len(data_dict["development"][f"{cur_emotion}_timeline"])):
        total_emotion += data_dict["development"][f"{cur_emotion}_timeline"][i]
        if i % (frame_rate*capture_interval) == 0 and i != 0:
            data_dict["data"][f"avg_{cur_emotion}_timeline"].append(total_emotion/i)
    data_dict["data"][f"overall_{cur_emotion}"] = round(total_emotion / len(data_dict["development"][f"{cur_emotion}_timeline"]), 2)

print(data_dict["development"]["stress_timeline"])
with open('result.json', 'w') as json_file:
    json.dump(data_dict, json_file)

fig, (ax1, ax2) = plt.subplots(2)
ax1.plot(range(len(data_dict["data"]["avg_stress_timeline"])), data_dict["data"]["avg_stress_timeline"])
ax1.set(xlabel="Session Duration (minutes)", ylabel="Avg. Stress Levels")
ax1.set_title("Stress Level Analysis Over Session Time (in Minutes)")


ax2.plot(range(len(data_dict["development"]["stress_timeline"])), data_dict["development"]["stress_timeline"])
ax2.set(xlabel="Session Duration (frames)", ylabel="Stress Levels")
ax2.set_title("Stress Level Analysis Over Session Time (in Frames)")

plt.tight_layout()
plt.show()


'''
img = cv2.imread("apeviaPowerSupply.jpg")
detector = FER(mtcnn=True)
detector.detect_emotions(img)
print(detector.emotions)
image = cv2.rectangle(img,
                      (detector.emotions[0]['box'][0], detector.emotions[0]['box'][1]),
                      (detector.emotions[0]['box'][0]+detector.emotions[0]['box'][2], detector.emotions[0]['box'][1]+detector.emotions[0]['box'][3]),
                      (254, 89, 194),
                      2)
font = cv2.FONT_HERSHEY_SIMPLEX
cv2.putText(img, f'Detected: {[max(detector.emotions[0]["emotions"], key=detector.emotions[0]["emotions"].get)]}',
            (detector.emotions[0]['box'][0], detector.emotions[0]['box'][1]-50), font, 1, (254, 89, 194), 2)

cv2.imshow("Image", image)
cv2.waitKey()
'''