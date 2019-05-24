from __future__ import print_function, division
import tensorflow as tf
import keras
from keras import applications, metrics, layers, models, regularizers, optimizers
from keras.applications import Xception, DenseNet201, InceptionV3
from keras.models import *
from keras.layers import *
import numpy as np
import time, cv2, os
from os.path import expanduser

# to get relative paths without using the user name
home = expanduser('~')

# threshold for the predictions. Increase to have less false positives
confidence_threshold = 0.5 

# radius threshold to differentiate between torx 6 and torx 8
torx6_radius_threshold = 15

# hough parameters, subject to change, depending on the height, illumination and etc.
hough_upper_threshold = 100
hough_lower_threshold = 30
hough_min_radius = 10
hough_max_radius = 30
        
# start the session
IMAGE_SIZE    = (71, 71)
NUM_CLASSES   = 2

base_model = Xception(include_top=False,
                        weights=None,
                        input_tensor=None,
                        input_shape=(IMAGE_SIZE[0],IMAGE_SIZE[1],3))
x = base_model.output
x = GlobalAveragePooling2D(name='avg_pool')(x)
x = Dropout(0.5)(x)
predictions = Dense(NUM_CLASSES, activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=predictions)

model.load_weights('model-final_x.h5', by_name=True)
#with tf.Session() as session:
imagenames = os.listdir('test_detection')
count  =1
file1 = open('det_2tf.txt', 'w')
try:
   os.mkdir('result_detection/')
except:
   pass

with tf.gfile.FastGFile(home + "/ownCloud/imagine_weights/screw_detector/retrained_graph.pb", 'rb') as f:
 
    graph_def = tf.GraphDef()	## The graph-graph_def is a saved copy of a TensorFlow graph; objektinitialisierung
    graph_def.ParseFromString(f.read())	#Parse serialized protocol buffer data into variable
    _ = tf.import_graph_def(graph_def, name='')	# import a serialized TensorFlow GraphDef protocol buffer, extract objects in the GraphDef as tf.Tensor

sess = tf.Session()
softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')
for name in imagenames:

        # flag we keep to know if we had any circles in the first place 
        noCircles = True

        # get a copy and resize it
        #img_raw = cv_image.copy()
        img_raw = cv2.imread('test_detection/'+name)
        
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
                cv2.rectangle(drawn_image, (x - r, y - r), (x + r, y + r), (255, 0, 0), 5)
                #cv2.imwrite("/home/eren/workspace/screw_detector/inception-network/training_dataset/hough/" + str(randint(1, 1000000)) + ".jpg", self.drawn_image)


                # get the above rectangle as ROI
                screw_roi = resized_img[y-r:y+r, x-r:x+r]
                #cv2.imwrite('extradata/'+str(count)+".jpg", screw_roi)
                count+=1
                #can't go on with the empty or corrupt roi
                if (screw_roi.size == 0):
                    break
                image_data = cv2.imencode('.jpg', screw_roi)[1].tostring()
                predictions = sess.run(softmax_tensor, {'DecodeJpeg/contents:0': image_data})

                image = cv2.resize(screw_roi, IMAGE_SIZE)
                np_image_data = np.asarray(image)
                np_image_data = np_image_data/255
                np_final = np.expand_dims(np_image_data,axis=0)
                aa = model.predict(np_final)
                
                score = (aa[0][1]+predictions[0][0])/2
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
sess.close()
file1.close()
