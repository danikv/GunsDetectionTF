import os
import cv2
import numpy as np
import tensorflow as tf
import sys
from queue import Queue
from threading import Thread
from remove_duplicates import remove_duplicates
from copy import deepcopy
from face_detection_main import face_det

# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")

from utils import label_map_util
from utils import visualization_utils as vis_util

class VideoCamera(object):
    def __init__(self):
        self.boxes = []
        self.classes = []
        self.scores = []
        self.counter = 0
        # Using OpenCV to capture from device 0. If you have trouble capturing
        # from a webcam, comment the line below out and use a video file
        # instead.
        self.video = cv2.VideoCapture('../guns_s.mp4')
        # self.vwriter = cv2.VideoWriter('vid.mp4', fps=25, frameSize=(1080,1920), fourcc=-1)
        # self.vwriter.open('vid.mp4', fps=25, frameSize=(1080,1920), fourcc=-1)
        # If you decide to use video.mp4, you must have this file in the folder
        # as the main.py.
        # self.video = cv2.VideoCapture('video.mp4')
        # Import utilites

        # Name of the directory containing the object detection module we're using
        MODEL_NAME = '../first_try_faster_rcnn_hackhacton'

        # Grab path to current working directory
        CWD_PATH = os.getcwd()

        # Path to frozen detection graph .pb file, which contains the model that is used
        # for object detection.
        PATH_TO_CKPT = os.path.join(CWD_PATH,MODEL_NAME,'frozen_inference_graph.pb')

        # Path to label map file
        PATH_TO_LABELS = os.path.join(CWD_PATH,'../config','oid_object_detection_label_map.pbtxt')

        # Number of classes the object detector can identify
        NUM_CLASSES = 4

        # Load the label map.
        # Label maps map indices to category names, so that when our convolution
        # network predicts `5`, we know that this corresponds to `king`.
        # Here we use internal utility functions, but anything that returns a
        # dictionary mapping integers to appropriate string labels would be fine
        self.label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
        self.categories = label_map_util.convert_label_map_to_categories(self.label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
        self.category_index = label_map_util.create_category_index(self.categories)

        # Load the Tensorflow model into memory.
        detection_graph = tf.Graph()
        with detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

            self.sess = tf.Session(graph=detection_graph)

        # Define input and output tensors (i.e. data) for the object detection classifier

        # Input tensor is the image
        self.image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

        # Output tensors are the detection boxes, scores, and classes
        # Each box represents a part of the image where a particular object was detected
        self.detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

        # Each score represents level of confidence for each of the objects.
        # The score is shown on the result image, together with the class label.
        self.detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
        self.detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')

        # Number of objects detected
        self.num_detections = detection_graph.get_tensor_by_name('num_detections:0')

        self.run_predictions = True

        self.gun_files_queue = Queue()
        self.faces_queue = Queue()

        self.remove_duplicates_thread = Thread(target=remove_duplicates, args=[self.gun_files_queue,
                                                                               ['static/images/references/gun_ref.jpg',
                                                                                'static/images/references/gun_ref2.jpg',
                                                                                'static/images/references/gun_ref3.jpg',
                                                                                'static/images/references/gun_ref4.jpg'
                                                                                ]])
        self.remove_duplicates_thread.start()

        self.faces_thread = Thread(target=face_det, args=[self.faces_queue])
        self.faces_thread.start()

    def get_predictions(self, ret, frame):
        if not ret:
            return None
        frame_expanded = np.expand_dims(frame, axis=0)
    
        # Perform the actual detection by running the model with the image as input
        (boxes, scores, classes, num) = self.sess.run(
            [self.detection_boxes, self.detection_scores, self.detection_classes, self.num_detections],
            feed_dict={self.image_tensor: frame_expanded})
        self.boxes = boxes[0]
        self.classes = classes[0]
        self.scores = scores[0]
        # print(frame.shape)
        for i, (box, score, clas) in enumerate(zip(self.boxes,self.scores,self.classes)):
            if clas == 1:
                if score >= 0.75:
                    # name = 'gun_fr{}_obj{}'.format(fr, id)
                    top = int(box[0] * frame.shape[0])
                    bot = int(box[2] * frame.shape[0])
                    left = int(box[1] * frame.shape[1])
                    right = int(box[3] * frame.shape[1])
                    # cv2.imwrite('{}.jpg'.format(name), frame[top:bot, left:right])
                    self.gun_files_queue.put(deepcopy(frame[top:bot, left:right]))
                    #id += 1
                # scores[i] += 0.15
                self.scores[i] = min(self.scores[i] + 0.05, 1)
            if clas == 4:
                self.scores[i] -= 0.15
            # if clas == 2:
            #     scores[i] -= 0.5
            # print(len(box))
            # print(clas)
            # print(score)
            # print(box)
    
    
        # Draw the results of the detection (aka 'visulaize the results')
        vis_util.visualize_boxes_and_labels_on_image_array(
            frame,
            np.squeeze(self.boxes),
            np.squeeze(self.classes).astype(np.int32),
            np.squeeze(self.scores),
            self.category_index,
            use_normalized_coordinates=True,
            line_thickness=8,
            min_score_thresh=0.80)
        return frame
    
    def __del__(self):
        self.video.release()
    
    def get_frame(self):
        success, image = self.video.read()
        # print(image.shape)
        self.counter += 1
        if self.counter % 1 == 0:
            self.run_predictions = True
            image = self.get_predictions(success, image)
        else :
            vis_util.visualize_boxes_and_labels_on_image_array(
                image,
                np.squeeze(self.boxes),
                np.squeeze(self.classes).astype(np.int32),
                np.squeeze(self.scores),
                self.category_index,
                use_normalized_coordinates=True,
                line_thickness=8,
                min_score_thresh=0.80)
            self.run_predictions = False
        if not success:
            # self.vwriter.release()
            return "".encode("utf-8")

        # cv2.imshow('img', image)
        # cv2.waitKey(1)
        # self.vwriter.write(image)
        if image is None or len(image) == 0:
            return "".encode("utf-8")
        ret, jpeg = cv2.imencode('.jpg', image)
        return jpeg.tobytes()
