import face_recognition
import cv2
import numpy as np
import constants
from remove_duplicates import get_ref_models_descriptors

def rect_to_css(rect):
    """
    Convert a dlib 'rect' object to a plain tuple in (top, right, bottom, left) order
    :param rect: a dlib 'rect' object
    :return: a plain tuple representation of the rect in (top, right, bottom, left) order
    """
    return rect.top(), rect.right(), rect.bottom(), rect.left()

def trim_css_to_bounds(css, image_shape):
    """
    Make sure a tuple in (top, right, bottom, left) order is within the bounds of the image.
    :param css:  plain tuple representation of the rect in (top, right, bottom, left) order
    :param image_shape: numpy shape of the image array
    :return: a trimmed plain tuple representation of the rect in (top, right, bottom, left) order
    """
    return max(css[0], 0), min(css[1], image_shape[1]), min(css[2], image_shape[0]), max(css[3], 0)

def face_recognition_event_loop(queue, ref_images):
    # Create arrays of known face encodings and their names
    known_face_encodings = []
    known_face_names = []

    orb = cv2.ORB_create()
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    ref_descriptors = get_ref_models_descriptors(orb, ref_images)
    # Initialize some variables
    face_locations = []
    face_encodings = []
    base_path = 'static/images/faces/'
    id = 0

    while True:
        ret = queue.get()
        if ret.all():
            break
        
        frame = cv2.resize(ret, (0, 0), fx=0.7, fy=0.7)
        small_frame = cv2.resize(frame, (0, 0), fx=1, fy=1)

        # Find all the faces and face encodings in the current frame of video
        face_locations = face_recognition.face_locations(small_frame, 1)
        face_encodings = face_recognition.face_encodings(small_frame, face_locations)

        for face_encoding, face_location in zip(face_encodings, face_locations):
            # See if the face is a match for the known face(s)
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            # Or instead, use the known face with the smallest distance to the new face

            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            if face_distances.size != 0:
                best_match_index = np.argmin(face_distances)
                if matches and matches[best_match_index]:
                    name, distance  = known_face_names[best_match_index]
                    (top, right, bottom, left) = face_location
                    kp1, des1 = orb.detectAndCompute(frame[top:bottom, left:right], None)
                    new_distance = image_distance_from_refs(ref_descriptors, des1, bf)
                    if new_distance > distance :
                        known_face_encodings[best_match_index] = face_encoding
                        known_face_names[best_match_index] = name, new_distance
                        write_image(face_location, base_path, name, frame)
                else:
                    name = "face{}".format(id)
                    id += 1
            else:
                name = "face{}".format(id)
                id += 1

            if True not in matches:
                known_face_encodings.append(face_encoding)
                (top, right, bottom, left) = face_location
                kp1, des1 = orb.detectAndCompute(frame[top:bottom, left:right], None)
                distance = image_distance_from_refs(ref_descriptors, des1, bf)
                known_face_names.append((name, distance))
                write_image(face_location, base_path, name, frame)
                
def write_image(face_location, base_path, name, image):
    (top, right, bottom, left) = face_location
    cv2.imwrite('{}/{}.jpg'.format(base_path, name), image[top:bottom, left:right])


def image_distance_from_refs(ref_descriptors, des1, bf):
    ref_matches = []
    for des in ref_descriptors:
        ref_matches.append(bf.match(des1, des))
    distance = 0
    for ref_match in ref_matches:
        filtered = [x.distance for x in ref_match if x.distance < constants.OUTLIERS_MIN_DIST]
        if not filtered:
            continue

        distance += sum(filtered) / len(filtered)
    return distance