import face_recognition
import cv2
import numpy as np
import copy

# This is a demo of running face recognition on live video from your webcam. It's a little more complicated than the
# other example, but it includes some basic performance tweaks to make things run a lot faster:
#   1. Process each video frame at 1/4 resolution (though still display it at full resolution)
#   2. Only detect faces in every other frame of video.

# PLEASE NOTE: This example requires OpenCV (the `cv2` library) to be installed only to read from your webcam.
# OpenCV is *not* required to use the face_recognition library. It's only required if you want to run this
# specific demo. If you have trouble installing it, try any of the other demos that don't require it instead.

# Get a reference to webcam #0 (the default one)
# video_capture = cv2.VideoCapture('guns.mp4')

# Load a sample picture and learn how to recognize it.
# obama_image = face_recognition.load_image_file("obama.jpg")
# obama_face_encoding = face_recognition.face_encodings(obama_image)[0]

# Load a second sample picture and learn how to recognize it.
# biden_image = face_recognition.load_image_file("biden.jpg")
# biden_face_encoding = face_recognition.face_encodings(biden_image)[0]

# Create arrays of known face encodings and their names
known_face_encodings = [
    # obama_face_encoding,
    # biden_face_encoding
]
known_face_names = [
    # "Barack Obama",
    # "Joe Biden"
]

# Initialize some variables
face_locations = []
face_encodings = []
face_names = []
process_this_frame = True
# frames = [face_recognition.load_image_file("friends.jpg"), face_recognition.load_image_file("friends2.jpg"),
#           face_recognition.load_image_file("how.jpg")]
# frame = face_recognition.load_image_file("friends.jpg")
# frame = cv2.imread("friends.jpg")
# cv2.imshow('ImageWindow', img)
# cv2.waitKey()
# cv2.imshow('img', frame)
# while True:
#     # Grab a single frame of video
#     ret, frame = video_capture.read()

# Resize frame of video to 1/4 size for faster face recognition processing
# small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

# Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
# rgb_small_frame = small_frame[:, :, ::-1]

# Only process every other frame of video to save time
id = 0
face_names = []
# for frame in frames:
framecount = 0
# while video_capture.isOpened():
#     ret, frame = video_capture.read()
frame = cv2.imread('exp.jpg')
# if not ret:
#     break
print(len(frame))
frame = cv2.resize(frame, (0, 0), fx=1, fy=1)
small_frame = cv2.resize(frame, (0, 0), fx=1, fy=1)
if process_this_frame:
    print("A")
    # Find all the faces and face encodings in the current frame of video
    face_locations = face_recognition.face_locations(small_frame)
    face_encodings = face_recognition.face_encodings(small_frame, face_locations)
    cpf = copy.deepcopy(frame)
    for face_encoding, face_location in zip(face_encodings, face_locations):
        # See if the face is a match for the known face(s)
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        # name = f"face{id}"
        # name = ""
        # # If a match was found in known_face_encodings, just use the first one.
        # if True in matches:
        #     first_match_index = matches.index(True)
        #     name = known_face_names[first_match_index]

        # Or instead, use the known face with the smallest distance to the new face

        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        if face_distances.size != 0:
            best_match_index = np.argmin(face_distances)
            if matches and matches[best_match_index]:
                name = known_face_names[best_match_index]
            else:
                name = f"face{id}"
                id += 1
        else:
            name = f"face{id}"
            id += 1

        if True not in matches:
            known_face_encodings.append(face_encoding)
            known_face_names.append(name)
            (top, right, bottom, left) = face_location
            # top *= 3
            # right *= 3
            # bottom *= 3
            # left *= 3
            cv2.imwrite(f'{name}.jpg', cv2.resize(cpf[top:bottom, left:right], (300,300), interpolation=cv2.INTER_LINEAR))

        (top, right, bottom, left) = face_location
        # top *= 3
        # right *= 3
        # bottom *= 3
        # left *= 3

        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # Draw a label with a name below the face
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
        if name and name not in face_names:
            face_names.append(name)

    cv2.imshow('img', frame)
    cv2.waitKey(1)

    # for (top, right, bottom, left), name in zip(face_locations, face_names):
        #     # Scale back up face locations since the frame we detected in was scaled to 1/4 size
        #     top *= 4
        #     right *= 4
        #     bottom *= 4
        #     left *= 4
        #
        #     # Draw a box around the face
        #     cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
        #
        #     # Draw a label with a name below the face
        #     cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        #     font = cv2.FONT_HERSHEY_DUPLEX
        #     cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

# framecount += 1
# process_this_frame = framecount % 10 == 0

# process_this_frame = not process_this_frame

# print(face_names)
# Display the results


# Display the resulting image
cv2.imwrite('faces.jpg', frame)
cv2.imshow('image', frame[:,:])
cv2.waitKey()
# # Hit 'q' on the keyboard to quit!
# if cv2.waitKey(1) & 0xFF == ord('q'):
#     break

# Release handle to the webcam
# video_capture.release()
cv2.destroyAllWindows()
