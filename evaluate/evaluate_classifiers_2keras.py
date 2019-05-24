from __future__ import print_function, division
import tensorflow as tf
import keras
from keras import applications, metrics, layers, models, regularizers, optimizers
from keras.applications import Xception, InceptionV3
from keras.models import *
from keras.layers import *
import numpy as np
import time, cv2, os

from os.path import expanduser

# to get relative paths without using the user name
home = expanduser('~')

IMAGE_SIZE1    = (71, 71)
NUM_CLASSES   = 2
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

model1 = MModel(home + '/ownCloud/imagine_weights/screw_detector/' + 'model-final_x.h5')
model2 = MModel2(home + '/ownCloud/imagine_weights/screw_detector/' + 'model-final_i.h5')


# radius threshold to differentiate between torx 6 and torx 8
torx6_radius_threshold = 15

# hough parameters, subject to change, depending on the height, illumination and etc.
hough_upper_threshold = 100
hough_lower_threshold = 25
hough_min_radius = 8
hough_max_radius = 20
        
imagenames = os.listdir(home + '/ownCloud/imagine_images/screw_data/test_detection/')
count  =1
file1 = open('det_2keras.txt', 'w')
try:
   os.mkdir('result_detection/')
except:
   pass

for name in imagenames:

        # flag we keep to know if we had any circles in the first place 
        noCircles = True

        # get a copy and resize it
        #img_raw = cv_image.copy()
        img_raw = cv2.imread(home + '/ownCloud/imagine_images/screw_data/test_detection/'+ name)
        
        img_h, img_w = img_raw.shape[:2]
        if img_h>img_w:
          ratiox = 986/img_w
          ratioy = 1382/img_h
          resized_img = cv2.resize(img_raw, (986,1382))
        else:
          ratiox = 1382/img_w
          ratioy = 986/img_h
          resized_img = cv2.resize(img_raw, (1382,986))

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
        circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 100, param1=hough_upper_threshold,param2=hough_lower_threshold,minRadius=hough_min_radius,maxRadius=hough_max_radius)
        # ensure at least some circles were found
        if circles is not None:
            noCircles = False
            # convert the (x, y) coordinates and radius of the circles to integers
            circles = np.round(circles[0, :]).astype("int")

            # copy the image, for painting we will use another
            drawn_image = resized_img.copy()
            #print("------------------------------------")
            # loop over the found circles
            for i in range(len(circles)):
                # get one
                (x, y, r) = circles[i]

                # draw the circle in the output image, then draw a rectangle corresponding to the center of the circle
                cv2.rectangle(drawn_image, (x - r, y - r), (x + r, y + r), (255, 0, 0), 2)

                # get the above rectangle as ROI
                screw_roi = resized_img[y-r:y+r, x-r:x+r]
                count+=1
                #can't go on with the empty or corrupt roi
                if (screw_roi.size == 0):
                    break

                image = cv2.resize(screw_roi, IMAGE_SIZE1)
                np_image_data = np.asarray(image)
                np_image_data = np_image_data/255
                np_final = np.expand_dims(np_image_data,axis=0)
                aa = model1.predict(np_final)
                
                image2 = cv2.resize(screw_roi, IMAGE_SIZE2)
                np_image_data2 = np.asarray(image2)
                np_image_data2 = np_image_data2/255
                np_final2 = np.expand_dims(np_image_data2,axis=0)
                aa2 = model2.predict(np_final2)

                score = (aa[0][1]+aa2[0][1])/2
                #print(aa)
                xmin = (x-r)/ratiox
                xmax = (x+r)/ratiox
                ymin = (y-r)/ratioy
                ymax = (y+r)/ratioy
                line_out = name.split('.')[0]
                #line_out += ' ' + str(aa[0][1]) + ' ' + str(xmin) + ' ' + str(ymin)+' ' + str(xmax) + ' ' + str(ymax) + '\n'
                line_out += ' ' + str(score) + ' ' + str(xmin) + ' ' + str(ymin)+' ' + str(xmax) + ' ' + str(ymax) + '\n'
                file1.write(line_out)
                if score>0.3:
                    cv2.circle(drawn_image, (int(x), int(y)), int(r), (0, 255, 0), 5) #green
        else:
            print("No detection of circles.")
            noCircles = True

        #return frame_data, noCircles
        cv2.imwrite('result_detection/'+name, drawn_image)
        #break
file1.close()
#frame_data, noCircles = detect_screw()
#cv2.imwrite('aa.jpg', frame_data)
