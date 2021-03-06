## DriveAID
DriveAID is a review/learning tool to help teen drivers improve their driving ability especially since covid-19 has had a significant impact in the education of new drivers.<br>
DriveAID aims to provide teens & parents with a method of monitoring their teen's driving ability through a concensual process in which both the teen and parent can review emotional and gesture data from a driving session to identify potential issues in their teen's driving.<br>
<p align="center">
<img width=400 src="https://user-images.githubusercontent.com/20238115/111500082-fb996b80-8719-11eb-9271-778065ac2b67.png" alt="DriveAID Project Logo"/>
</p>

### Demo
[Emotion And Gesture Recognition Demo](https://www.youtube.com/watch?v=2JThDjU4U14)

### Presentation
[Makeathon Presentation](https://github.com/TeamInnovAIte/InnovAIte/blob/main/TeamInnovAIte%20-%20Makeathon%20Presentation.pptx)


### Requirements
Each detection model folder contains a `requirements.txt` file. Please install the dependencies inside using `pip`<br>
```
pip install -R emotion_detection/requirements.txt
pip install -R gesture_detection/requirements.txt
```
The dashboard demo requirements can be found in the `package.json` file.


### Additional Datasets
[Dlib Shape Predictor 68 Face Landmarks](https://github.com/davisking/dlib-models/blob/master/shape_predictor_68_face_landmarks.dat.bz2)


### Licenses
[Library - FER](https://github.com/justinshenk/fer/blob/master/LICENSE)
- This project utilizes the FER (facial emotion recognition) library's CNN facial detection model. The usage of this library has been listed here as per the MIT licensing agreement.

[Graphics - Google Material Design Icons](https://github.com/google/material-design-icons/blob/master/LICENSE)
- The logo for our project utilizes custom modified icons from the Google Material Design Icons library. This change has been listed here as per the Apache 2.0 licensing agreement of the library.

### References and Articles
[GitHub - Geek-ubaid/Stress-Detection](https://github.com/Geek-ubaid/Stress-Detection)<br>
[CNN Based Face Detector From DLIB](https://towardsdatascience.com/cnn-based-face-detector-from-dlib-c3696195e01c
)<br>
[Face Detection Using OpenCV](https://towardsdatascience.com/face-detection-in-2-minutes-using-opencv-python-90f89d7c0f81)<br>
[CNN State Farm Distracted Driver Detection Notebook](https://www.kaggle.com/anayantzinp/cnn-state-farm-distracted-driver-detection)<br>
[CNN State Farm Distracted Driver Detection Dataset](https://www.kaggle.com/c/state-farm-distracted-driver-detection/overview)<br>
[Affectiva In-Cabin Sensing For Teen Driver Safety](https://blog.affectiva.com/how-in-cabin-sensing-helps-teen-drivers-stay-safe)
