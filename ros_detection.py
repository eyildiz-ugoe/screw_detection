#!/usr/bin/env python
from __future__ import print_function, division
import cv2
import numpy as np
import argparse
import lazyros.basics
import image_geometry
import rospy
import tensorflow as tf
import sys
import os
from os.path import expanduser
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from random import *
import matplotlib.pyplot as plt
from dynamic_reconfigure.server import Server
from screw_detection.cfg import ScrewDetectionConfig
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure

from keras.applications import Xception, InceptionV3
from keras.models import *
from keras.layers import *
import numpy as np

# Disable tensorflow compilation warnings
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

# to get relative paths without using the user name
home = expanduser('~')

# user specific path, change this to your dir_to_the_data path, folders in the path must exist beforehand
dir_to_the_data = '/ownCloud/imagine_images/screw_data/train1/all/'
dir_to_the_models = '/ownCloud/imagine_weights/screw_detector/'

# screw and artifact
NUM_CLASSES   = 2

# class for the first NN of Xception
IMAGE_SIZE1    = (71, 71)
class MModel:    
    @staticmethod
    def loadmodel(path):
        base_model1 = Xception(include_top=False,
                        weights=None,
                        input_tensor=None,
                        input_shape=(IMAGE_SIZE1[0],IMAGE_SIZE1[1],3))
        x1 = base_model1.output
        x1 = GlobalAveragePooling2D(name='avg_pool')(x1)
        x1 = Dropout(0.5)(x1)
        predictions1 = Dense(NUM_CLASSES, activation='softmax')(x1)
        model1 = Model(inputs=base_model1.input, outputs=predictions1)
        model1.load_weights(path)
        return model1
  
    def __init__(self, path):
       self.model = self.loadmodel(path)
       self.graph = tf.get_default_graph()

    def predict(self, X):
        with self.graph.as_default():
            return self.model.predict(X)
    
# class for the first NN of InceptionV3
IMAGE_SIZE2    = (139, 139)
class MModel2:    
    @staticmethod
    def loadmodel(path):
        base_model1 = InceptionV3(include_top=False,
                        weights=None,
                        input_tensor=None,
                        input_shape=(IMAGE_SIZE2[0],IMAGE_SIZE2[1],3))
        x1 = base_model1.output
        x1 = GlobalAveragePooling2D(name='avg_pool')(x1)
        x1 = Dropout(0.5)(x1)
        predictions1 = Dense(NUM_CLASSES, activation='softmax')(x1)
        model1 = Model(inputs=base_model1.input, outputs=predictions1)
        model1.load_weights(path)
        return model1
  
    def __init__(self, path):
       self.model = self.loadmodel(path)
       self.graph = tf.get_default_graph()

    def predict(self, X):
        with self.graph.as_default():
            return self.model.predict(X)

class ScrewDetector:
    def __init__(self):
        self.dyn_reconf_server = Server(ScrewDetectionConfig, self.callback_reconf) 
        self.bridge = CvBridge()
        self.received_img = None # an internal variable needed for image acquisition
        self.cv_image = None # an internal variable needed for opencv-ros image conversion
        self.result_image_pub = rospy.Publisher("result_image", Image, queue_size=1) # to display the result of the detection on rviz

        # mode identifier: offline mode = 0 - to collect data,  online mode = 1 - to run the detection
        self.mode = 1
       
        # camera instrinsics and camera model
        self.camera_model = image_geometry.PinholeCameraModel()
        self.camera_info = lazyros.basics.receive_one_message("/pylon_camera_node/camera_info")
        self.camera_model.fromCameraInfo(self.camera_info)

        # make the original image's resolution this smaller
        self.resizing_factor = 0.3 

        # threshold for the predictions. Increase to have less false positives
        self.confidence_threshold = 0.8 

        # radius threshold to differentiate between torx 6 and torx 8
        self.torx6_radius_threshold = 5

        # hough parameters, subject to change, depending on the height, illumination and etc.
        self.hough_upper_threshold = 100
        self.hough_lower_threshold = 50
        self.hough_min_radius = 5
        self.hough_max_radius = 30

        # to keep the singular and multiple screw data
        self.screw = dict()
        self.screw_data = []
        
        # define keras models
        self.model1 = MModel(home + dir_to_the_models + 'model-final_x.h5')
        self.model2 = MModel2(home + dir_to_the_models + 'model-final_i.h5')

        # get the image from basler camera
        self.image_sub = rospy.Subscriber("/pylon_camera_node/image_raw", Image, self.image_cb)

    # re-adjust the values depending on the GUI inputs
    def callback_reconf(self, config, level):
        rospy.loginfo("Config set to {resizing_factor}, {screw_detection_confidence_threshold}, {torx6_radius_threshold}, {hough_upper_threshold}, {hough_lower_threshold}, {hough_min_radius}, {hough_max_radius}, {mode}".format(**config))

        self.resizing_factor = config.resizing_factor
        self.confidence_threshold = config.screw_detection_confidence_threshold
        self.torx6_radius_threshold = config.torx6_radius_threshold
        self.hough_upper_threshold = config.hough_upper_threshold
        self.hough_lower_threshold = config.hough_lower_threshold
        self.hough_min_radius = config.hough_min_radius
        self.hough_max_radius = config.hough_max_radius
        self.mode = config.mode

        return config

    def image_cb(self, msg):
        # image is received 
        self.received_img = msg
        try:
            self.cv_image = self.bridge.imgmsg_to_cv2(self.received_img, "bgr8")
            self.screw_data, self.drawn_image = self.collect_or_detect_screw() # conduct the detection  

            #  publish the result, which is an image (scaled) 
            result_image_msg = Image()
            try:
                result_image_msg = self.bridge.cv2_to_imgmsg(self.drawn_image, "bgr8") 
                #print(self.screw_data)
            except CvBridgeError as e:
                print("Could not make it through the cv bridge of death.")

            self.result_image_pub.publish(result_image_msg)
        
            # reset the screw data
            self.screw_data = []

        except CvBridgeError as e:
            print("Monocular could not make it through the cv bridge of death.")

    def collect_or_detect_screw(self):

        # flag we keep to know if we had any circles in the first place 
        noCircles = True

        # get a copy and resize it
        img_raw = self.cv_image.copy()
        resized_img = cv2.resize(img_raw, (0,0), fx=self.resizing_factor, fy=self.resizing_factor)

        # grayscale it
        gray = cv2.cvtColor(resized_img, cv2.COLOR_BGR2GRAY)

        # detect circles in the image
        """
        Parameters:	

        image: 8-bit, single-channel, grayscale input image.
        circles: Output vector of found circles. Each vector is encoded as a 3-element floating-point vector (x, y, radius).
        method: Always CV_HOUGH_GRADIENT.
        dp: Always 1.
        minDist: Minimum distance between the centers of the detected circles. If the parameter is too small, multiple neighbor circles may be falsely detected in addition to a true one. If it is too large, some circles may be missed.
        param1: First method-specific parameter. It is the higher threshold of the two passed to the Canny() edge detector (the lower one is twice smaller).
        param2: Second method-specific parameter. It is the accumulator threshold for the circle centers at the detection stage. The smaller it is, the more false circles may be detected. Circles, corresponding to the larger accumulator values, will be returned first.
        minRadius: Minimum circle radius.
        maxRadius: Maximum circle radius.
        """
        circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 100, param1=self.hough_upper_threshold,param2=self.hough_lower_threshold,minRadius=self.hough_min_radius,maxRadius=self.hough_max_radius)
        # ensure at least some circles were found
        if circles is not None:
            noCircles = False
            # convert the (x, y) coordinates and radius of the circles to integers
            circles = np.round(circles[0, :]).astype("int")

            # copy the image, for painting we will use another
            self.drawn_image = resized_img.copy()
            # loop over the found circles
            for i in range(len(circles)):
                # get one
                (x, y, r) = circles[i]

                # draw the circle in the output image, then draw a rectangle corresponding to the center of the circle
                cv2.rectangle(self.drawn_image, (x - r, y - r), (x + r, y + r), (255, 0, 0), 5)

                # get the above rectangle as ROI
                screw_roi = resized_img[y-r:y+r, x-r:x+r]
                
                # if the mode is data collection mode
                if (self.mode == 0):
                    cv2.imwrite(home + dir_to_the_data + str(randint(1, 1000000)) + ".jpg", screw_roi)
                    print("Data Collection Mode: On. Saving candidates under: " + home + dir_to_the_data)
                else:
                    print("Detection Mode: On. Using models under: " + home + dir_to_the_models)
                    # can't go on with the empty or corrupt roi
                    if (screw_roi.size == 0):
                        break
                    
                    # use dcnns and classify whether or not the ROI is a screw or not
                    image = cv2.resize(screw_roi, IMAGE_SIZE1)
                    np_image_data = np.asarray(image)
                    np_image_data = np_image_data/255
                    np_final = np.expand_dims(np_image_data,axis=0)
                    predicted1 = self.model1.predict(np_final)
                    
                    image2 = cv2.resize(screw_roi, IMAGE_SIZE2)
                    np_image_data2 = np.asarray(image2)
                    np_image_data2 = np_image_data2/255
                    np_final2 = np.expand_dims(np_image_data2,axis=0)
                    predicted2 = self.model2.predict(np_final2)
                    score = predicted1[0][1]+predicted2[0][1]
                    
                    # if it crosses the confidence threshold
                    if (score > self.confidence_threshold):
                        # now decide on the type, based on the radius
                        if (r > self.torx6_radius_threshold):
                            part_type = 'torx8'
                            # mark the screw
                            cv2.circle(self.drawn_image, (x, y), r, (0, 255, 0), 4) #green
                        else:
                            part_type = 'torx6'
                            # mark the screw
                            cv2.circle(self.drawn_image, (x, y), r, (0, 0, 255), 4) #red

                        # remap in the original image
                        scaled_point = (round(x * (1/self.resizing_factor)), round(y * (1/self.resizing_factor)))

                        # append to the dict
                        self.screw[scaled_point] = [part_type, "screw" + str(i)]

                        # record the data into frame dictionary
                        self.screw_data.append(self.screw)
        else:
            print("No detection of circles.")
            noCircles = True

        return self.screw_data, self.drawn_image

if __name__ == "__main__":
    rospy.init_node("screw_detection")
    wrapper = ScrewDetector()
    print("Running ...")
    rospy.spin()
    print("Done")
