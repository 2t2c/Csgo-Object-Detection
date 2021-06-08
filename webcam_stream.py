import time
import cv2
import mss
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import sys
import tensorflow as tf
from distutils.version import StrictVersion
from collections import defaultdict
from io import StringIO
import datetime
import multiprocessing
from multiprocessing import Pipe
from threading import Thread


def Webcam():
    cap = cv2.VideoCapture(0)
    # cap.set(3, 1280) ; 3 for width
    # cap.set(4, 720) ; 4 for height

    while (True):
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)
        if ret == True:
            font = cv2.FONT_HERSHEY_SIMPLEX
            datet = str(datetime.datetime.now())
            frame = cv2.putText(frame, datet, (10, 50), font, 0.7, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.imshow("Frame", frame)
            if cv2.waitKey(1) & 0xFF == ord("e"):
                break
        else:
            break

    cap.release()
    cv2.destroyAllWindows()


# Env setup
from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

# # Model preparation
PATH_TO_FROZEN_GRAPH = 'X:/TfObjectDetection/research/object_detection/data/frozen_inference_graph.pb'
# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = 'X:/TfObjectDetection/research/object_detection/data/mscoco_label_map.pbtxt'
NUM_CLASSES = 99

# ## Load a (frozen) Tensorflow model into memory.
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES,
                                                            use_display_name=True)
category_index = label_map_util.create_category_index(categories)

detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.compat.v1.GraphDef()
    with tf.compat.v2.io.gfile.GFile(PATH_TO_FROZEN_GRAPH, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

# Detection
def TensorflowDetection():
    title = "FPS benchmark"
    start_time = time.time()
    display_time = 2
    fps = 0
    sct = mss.mss()
    monitor = {"top": 40, "left": 0, "width": 800, "height": 640}
    with detection_graph.as_default():
        with tf.compat.v1.Session(graph=detection_graph) as sess:
            while True:
                # Get raw pixels from the screen, save it to a Numpy array
                image_np = np.array(sct.grab(monitor))
                # To get real color we do this:
                image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
                # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
                image_np_expanded = np.expand_dims(image_np, axis=0)
                # Actual detection.
                image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
                boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
                scores = detection_graph.get_tensor_by_name('detection_scores:0')
                classes = detection_graph.get_tensor_by_name('detection_classes:0')
                num_detections = detection_graph.get_tensor_by_name('num_detections:0')
                # Visualization of the results of a detection.
                (boxes, scores, classes, num_detections) = sess.run(
                    [boxes, scores, classes, num_detections],
                    feed_dict={image_tensor: image_np_expanded})
                vis_util.visualize_boxes_and_labels_on_image_array(
                    image_np,
                    np.squeeze(boxes),
                    np.squeeze(classes).astype(np.int32),
                    np.squeeze(scores),
                    category_index,
                    use_normalized_coordinates=True,
                    line_thickness=3)
                # Show image with detection
                cv2.imshow(title, cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB))
                # Bellow we calculate our FPS
                fps += 1
                TIME = time.time() - start_time
                if (TIME) >= display_time:
                    print("FPS: ", fps / (TIME))
                    fps = 0
                    start_time = time.time()
                # Press "q" to quit
                if cv2.waitKey(25) & 0xFF == ord("q"):
                    cv2.destroyAllWindows()
                    break

if __name__=="__main__":
    t1 = Thread(target=Webcam)
    t2 = Thread(target=TensorflowDetection)

    t1.start()
    t2.start()
