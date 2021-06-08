import time
import cv2
import mss
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import sys
import tensorflow as tf
from distutils.version import StrictVersion
from collections import defaultdict
from io import StringIO
import pyautogui
import pydirectinput
import datetime
import math
from threading import Thread
from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

# # Model preparation
PATH_TO_FROZEN_GRAPH = 'CSGO_frozen_inference_graph.pb'
# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = 'CSGO_labelmap.pbtxt'
NUM_CLASSES = 4

# Load a (frozen) Tensorflow model into memory.
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES,
                                                            use_display_name=True)
category_index = label_map_util.create_category_index(categories)

detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.compat.v1.GraphDef()
    with tf.compat.v1.gfile.GFile(PATH_TO_FROZEN_GRAPH, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')


def Shoot(mid_x, mid_y):
    width = 800
    height = 600
    x = int(mid_x * width)
    y = int(mid_y * height + height / 9)
    # print(x,y)
    # pydirectinput.moveTo(x, y)
    pydirectinput.click()

def Gesture():
    cap = cv2.VideoCapture(0)
    while (cap.isOpened()):
        # read image
        width = 800
        height = 600
        ret, img = cap.read()
        img = cv2.flip(img, 1)
        # get hand data from the rectangle sub window on the screen
        cv2.rectangle(img, (0, 1), (300, 500), (0, 255, 0), 0)
        crop_img = img[1:500, 0:300]

        # convert to grayscale
        grey = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)

        # applying gaussian blur
        ksize = (35, 35)
        blurred = cv2.GaussianBlur(grey, ksize, 0)

        # thresholdin: Otsu's Binarization method
        _, thresh1 = cv2.threshold(blurred, 140, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # show thresholded image
        # cv2.imshow('Thresholded', thresh1)

        # check OpenCV version to avoid unpacking error
        (version, _, _) = cv2.__version__.split('.')

        if version == '3':
            image, contours, hierarchy = cv2.findContours(thresh1.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        elif version == '4':
            contours, hierarchy = cv2.findContours(thresh1.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # find contour with max area
        cnt = max(contours, key=lambda x: cv2.contourArea(x))

        # create bounding rectangle around the contour (can skip below two lines)
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(crop_img, (x, y), (x + w, y + h), (0, 0, 255), 0)

        # finding convex hull
        hull = cv2.convexHull(cnt)

        # drawing contours
        drawing = np.zeros(crop_img.shape, np.uint8)
        cv2.drawContours(drawing, [cnt], 0, (0, 255, 0), 0)
        cv2.drawContours(drawing, [hull], 0, (0, 0, 255), 0)

        # finding convex hull
        hull = cv2.convexHull(cnt, returnPoints=False)

        # finding convexity defects
        defects = cv2.convexityDefects(cnt, hull)
        count_defects = 0
        cv2.drawContours(thresh1, contours, -1, (0, 255, 0), 3)

        # applying Cosine Rule to find angle for all defects (between fingers)
        # with angle > 90 degrees and ignore defects
        for i in range(defects.shape[0]):
            s, e, f, d = defects[i, 0]

            start = tuple(cnt[s][0])
            end = tuple(cnt[e][0])
            far = tuple(cnt[f][0])

            # find length of all sides of triangle
            a = math.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)
            b = math.sqrt((far[0] - start[0]) ** 2 + (far[1] - start[1]) ** 2)
            c = math.sqrt((end[0] - far[0]) ** 2 + (end[1] - far[1]) ** 2)

            # apply cosine rule here
            angle = math.acos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c)) * 57

            # ignore angles > 90 and highlight rest with red dots
            if angle <= 90:
                count_defects += 1
                cv2.circle(crop_img, far, 1, [0, 0, 255], -1)
            # dist = cv2.pointPolygonTest(cnt,far,True)

            # draw a line from start to end i.e. the convex points (finger tips)
            cv2.line(crop_img, start, end, [0, 255, 0], 2)

        # define actions required
        if count_defects == 1:
            cv2.putText(img, "Forward", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            pydirectinput.press('w')
        elif count_defects == 2:
            cv2.putText(img, "Left", (5, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            pydirectinput.press('a')
        elif count_defects == 3:
            cv2.putText(img, "Right", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            pydirectinput.press('s')
        elif count_defects == 4:
            cv2.putText(img, "Backward", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            pydirectinput.press('d')
        else:
            cv2.putText(img, "Jump", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            pydirectinput.press('space')

        # show appropriate images in windows
        cv2.imshow('Gesture', img)
        # all_img = np.hstack((drawing, crop_img))
        # cv2.imshow('Contours', all_img)

        k = cv2.waitKey(10)
        if k == 27:
            break

# Detection
def TensorflowDetection():
    title = "FPS benchmark"
    start_time = time.time()
    display_time = 2
    fps = 0
    sct = mss.mss()
    width = 800
    height = 600
    monitor = {"top": 80, "left": 0, "width": width, "height": height}
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
                    line_thickness=2)

                array_ch = []
                array_c = []
                array_th = []
                array_t = []
                for i, b in enumerate(boxes[0]):
                    if classes[0][i] == 2:  # ch
                        if scores[0][i] >= 0.5:
                            mid_x = (boxes[0][i][1] + boxes[0][i][3]) / 2
                            mid_y = (boxes[0][i][0] + boxes[0][i][2]) / 2
                            array_ch.append([mid_x, mid_y])
                            cv2.circle(image_np, (int(mid_x * width), int(mid_y * height)), 3, (0, 0, 255), -1)
                    if classes[0][i] == 1:  # c
                        if scores[0][i] >= 0.5:
                            mid_x = (boxes[0][i][1] + boxes[0][i][3]) / 2
                            mid_y = boxes[0][i][0] + (boxes[0][i][2] - boxes[0][i][0]) / 6
                            array_c.append([mid_x, mid_y])
                            cv2.circle(image_np, (int(mid_x * width), int(mid_y * height)), 3, (50, 150, 255), -1)
                    if classes[0][i] == 4:  # th
                        if scores[0][i] >= 0.5:
                            mid_x = (boxes[0][i][1] + boxes[0][i][3]) / 2
                            mid_y = (boxes[0][i][0] + boxes[0][i][2]) / 2
                            array_th.append([mid_x, mid_y])
                            cv2.circle(image_np, (int(mid_x * width), int(mid_y * height)), 3, (0, 0, 255), -1)
                    if classes[0][i] == 3:  # t
                        if scores[0][i] >= 0.5:
                            mid_x = (boxes[0][i][1] + boxes[0][i][3]) / 2
                            mid_y = boxes[0][i][0] + (boxes[0][i][2] - boxes[0][i][0]) / 6
                            array_t.append([mid_x, mid_y])
                            cv2.circle(image_np, (int(mid_x * width), int(mid_y * height)), 3, (50, 150, 255), -1)

                enemy_team = "t"  # shooting target
                if enemy_team == "c":
                    if len(array_ch) > 0:
                        Shoot(array_ch[0][0], array_ch[0][1])
                    if len(array_ch) == 0 and len(array_c) > 0:
                        Shoot(array_c[0][0], array_c[0][1])
                elif enemy_team == "t":
                    if len(array_th) > 0:
                        Shoot(array_th[0][0], array_th[0][1])
                    if len(array_th) == 0 and len(array_t) > 0:
                        Shoot(array_t[0][0], array_t[0][1])

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
    t1 = Thread(target=Gesture)
    t2 = Thread(target=TensorflowDetection)

    t1.start()
    t2.start()