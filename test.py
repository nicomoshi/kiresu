import face_recognition
import cv2
import pyrebase
import requests
import numpy as np
from sklearn.metrics import pairwise
from keras.models import load_model
import firebase_admin
from firebase_admin import credentials, firestore

cred = credentials.Certificate("firebase-cred.json")
firebase_admin.initialize_app(cred)

db = firestore.client()


def predict_rgb_image_vgg(image):
    image = np.array(image, dtype='float32')
    image /= 255
    pred_array = model.predict(image)
    result = gesture_names[np.argmax(pred_array)]
    score = float("%0.2f" % (max(pred_array[0]) * 100))
    return result, score

def remove_background(frame):
    fgmask = bgModel.apply(frame, learningRate=learningRate)
    kernel = np.ones((3, 3), np.uint8)
    fgmask = cv2.erode(fgmask, kernel, iterations=1)
    res = cv2.bitwise_and(frame, frame, mask=fgmask)
    return res

# General  Hand Recognition Settings
prediction = ''
action = ''
score = 0
img_counter = 500

# Gestures
gesture_names = {0: 'Fist',
                 1: 'L',
                 2: 'Okay',
                 3: 'Palm',
                 4: 'Peace'}

gesture_emojis = {
    'Fist': '✊',
    'L': '👆',
    'Okay': '👌',
    'Palm': '🤚',
    'Peace': '✌️'
}

# Hand Recognition Model
model = load_model('models/VGG_cross_validated.h5')

# parameters
cap_region_x_begin = 0.75  # start point/total width
cap_region_y_end = 0.6  # start point/total width
threshold = 1  # binary threshold
blurValue = 41  # GaussianBlur parameter
bgSubThreshold = 100
learningRate = 0

# variableslt
isBgCaptured = 0  # bool, whether the background captured
triggerSwitch = False  # if true, keyboard simulator works

# users collection
snapshots = list(db.collection(u'users').get())

# number of users
number_of_images = 0
for snapshot in snapshots:
    number_of_images += 1

images = [] * (number_of_images+1)  # array for image storing
encodings = [] * (number_of_images+1)  # array for image encodings

known_face_names = []  # known faces
known_gestures = []

snapshot_controller = 0


# Start camera
video_capture = cv2.VideoCapture(0)

for snapshot in snapshots:
    get_name = snapshot.to_dict()["fName"]
    response = requests.get(snapshot.to_dict()["photo_url"])
    r = open(get_name + ".png", "wb")
    r.write(response.content)
    r.close()
    images.append(face_recognition.load_image_file(get_name + ".png"))
    encodings.append(face_recognition.face_encodings(images[snapshot_controller])[0])  #Fix
    known_face_names.append(get_name)
    known_gestures.append(snapshot.to_dict()["gesture"])
    snapshot_controller += 1

# Initialize some variables
face_locations = []
face_encodings = []
face_names = []
process_this_frame = True

gesture_timer = 10

while True:
    # Grab a single frame of video
    ret, frame = video_capture.read()

    frame = cv2.bilateralFilter(frame, 5, 50, 100)  # smoothing filter
    frame = cv2.flip(frame, 1)  # flip the frame horizontally
    # Resize frame of video to 1/4 size for faster face recognition processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    rgb_small_frame = small_frame[:, :, ::-1]
    cv2.rectangle(frame, (int(cap_region_x_begin * frame.shape[1]), 0),
                  (frame.shape[1], int(cap_region_y_end * frame.shape[0])), (255, 0, 0), 2)

    bgModel = cv2.createBackgroundSubtractorMOG2(0, bgSubThreshold)

    img = remove_background(frame)
    img = img[0:int(cap_region_y_end * frame.shape[0]),
          int(cap_region_x_begin * frame.shape[1]):frame.shape[1]]  # clip the ROI

    # convert the image into binary image
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (blurValue, blurValue), 0)
    capture = 0

    # cv2.imshow('blur', blur)
    ret, thresh = cv2.threshold(blur, threshold, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # Add prediction and action text to thresholded image
    # cv2.putText(thresh, f"Prediction: {prediction} ({score}%)", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255))
    # cv2.putText(thresh, f"Action: {action}", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255))  # Draw the text
    # Draw the text
    cv2.imshow('ori', thresh)
    cv2.putText(frame, f"{prediction} ({score}%)", (1000, 400), cv2.FONT_HERSHEY_SIMPLEX, 1,
                (0, 0, 0))

    # copies 1 channel BW image to all 3 RGB channels
    target = np.stack((thresh,) * 3, axis=-1)
    target = cv2.resize(target, (224, 224))
    target = target.reshape(1, 224, 224, 3)
    prediction, score = predict_rgb_image_vgg(target)
    # Only process every other frame of video to save time
    if process_this_frame:
        # Find all the faces and face encodings in the current frame of video
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = []
        gestures = []
        for face_encoding in face_encodings:
            # See if the face is a match for the known face(s)
            matches = face_recognition.compare_faces(encodings, face_encoding)
            name = "Unknown"

            # If a match was found in known_face_encodings, just use the first one.
            if True in matches:
                first_match_index = matches.index(True)
                name = known_face_names[first_match_index]
                gesture = known_gestures[first_match_index]

            face_names.append(name)
            gestures.append(gesture)

    process_this_frame = not process_this_frame


    # Display the results
    for (top, right, bottom, left), name, gesture in zip(face_locations, face_names, gestures):
        # Scale back up face locations since the frame we detected in was scaled to 1/4 size
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # Draw a label with a name below the face
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

        if gesture_emojis[prediction] == gesture:
            if score > 95:
                gesture_timer -= 1
                if gesture_timer == 0:
                    print('door open')
            else:
                gesture_timer = 10

    cv2.imshow('original', frame)

    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()

