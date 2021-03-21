import face_recognition
import cv2
import pyrebase
import requests
import numpy as np
from sklearn.metrics import pairwise

# Firebase config
config = {
  "apiKey": "AIzaSyCL6_tL1MRh4cdwk7QjlogFvlppvIsTM1E",
  "authDomain": "kiresu-6d765.firebaseapp.com",
  "databaseURL": "https://kiresu-6d765-default-rtdb.firebaseio.com",
  "storageBucket": "kiresu-6d765.appspot.com"
}

# Firebase connection
firebase = pyrebase.initialize_app(config)

# Firebase database instance
db = firebase.database()

# Get one name and image
name = db.child("users").child("Rodolfo").get().val().get('name')
image = db.child("users").child("Rodolfo").get().val().get('image')

# Download image
response = requests.get(image)
r = open("Rodolfo.png", "wb")
r.write(response.content)
r.close()

# Start camera
video_capture = cv2.VideoCapture(0)

images = []
images.append(face_recognition.load_image_file('Rodolfo.png'))
encodings = []
encodings.append(face_recognition.face_encodings(images[0])[0])
known_face_names = ['Rodolfo']


# Initialize some variables
face_locations = []
face_encodings = []
face_names = []
process_this_frame = True

i = 0
_, frame=video_capture.read()
back = None
roi = cv2.selectROI(frame)
(x, y, w, h) = tuple(map(int, roi))

while True:
    # Grab a single frame of video
    ret, frame = video_capture.read()

    # Resize frame of video to 1/4 size for faster face recognition processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    rgb_small_frame = small_frame[:, :, ::-1]

    # Only process every other frame of video to save time
    if process_this_frame:
        # Find all the faces and face encodings in the current frame of video
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = []
        for face_encoding in face_encodings:
            # See if the face is a match for the known face(s)
            matches = face_recognition.compare_faces(encodings, face_encoding)
            name = "Unknown"

            # If a match was found in known_face_encodings, just use the first one.
            if True in matches:
                first_match_index = matches.index(True)
                name = known_face_names[first_match_index]

            face_names.append(name)

    process_this_frame = not process_this_frame


    # Display the results
    for (top, right, bottom, left), name in zip(face_locations, face_names):
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

    if i < 60:
        i += 1
        if back is None:
            back = frame[y:y + h, x:x + w].copy()
            back = np.float32(back)
        else:

            cv2.accumulateWeighted(frame[y:y + h, x:x + w].copy(), back, 0.2)
    else:
        # print(back.shape,frame.shape)
        back = cv2.convertScaleAbs(back)
        back_gray = cv2.cvtColor(back, cv2.COLOR_BGR2GRAY)
        frame_gray = cv2.cvtColor(frame[y:y + h, x:x + w], cv2.COLOR_BGR2GRAY)

        img = cv2.absdiff(back_gray, frame_gray)

        ret, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        con, hie = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        img2 = img.copy()

        con = max(con, key=cv2.contourArea)
        conv_hull = cv2.convexHull(con)
        cv2.drawContours(img, [conv_hull], -1, 225, 3)

        top = tuple(conv_hull[conv_hull[:, :, 1].argmin()][0])
        bottom = tuple(conv_hull[conv_hull[:, :, 1].argmax()][0])
        left = tuple(conv_hull[conv_hull[:, :, 0].argmin()][0])
        right = tuple(conv_hull[conv_hull[:, :, 0].argmax()][0])
        cx = (left[0] + right[0]) // 2
        cy = (top[1] + bottom[1]) // 2

        dist = pairwise.euclidean_distances([left, right, bottom, top], [[cx, cy]])[0]
        radi = int(0.80 * dist)

        circular_roi = np.zeros_like(img, dtype='uint8')
        cv2.circle(circular_roi, (cx, cy), radi, 255, 8)
        wighted = cv2.addWeighted(img.copy(), 0.6, circular_roi, 0.4, 2)

        mask = cv2.bitwise_and(img2, img2, mask=circular_roi)
        # mask
        con, hie = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        count = 0
        circumfrence = 2 * np.pi * radi
        for cnt in con:
            (m_x, m_y, m_w, m_h) = cv2.boundingRect(cnt)
            out_wrist_range = (cy + (cy * 0.25)) > (m_y + m_h)
            limit_pts = (circumfrence * 0.25) > cnt.shape[0]
            if limit_pts and out_wrist_range:
                # print(limit_pts,out_wrist_range)
                count += 1

        cv2.putText(frame, 'count: ' + str(count), (460, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 250, 0), thickness=4)
        cv2.rectangle(frame, (x, y), (x + w, y + h), 255, 3)
        cv2.imshow('mask', mask)
        cv2.imshow('frame', frame)
        cv2.imshow('weight', wighted)


    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()