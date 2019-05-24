from __future__ import print_function, division
import os.path
import keras
from keras import applications, metrics, layers, models, regularizers, optimizers
from keras.applications import InceptionV3, Xception
from keras.models import *
from keras.layers import *
import numpy as np
import time, cv2, os
from os.path import expanduser

# to get relative paths without using the user name
home = expanduser('~')

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

model.load_weights(home + '/ownCloud/imagine_weights/screw_detector/pretrained_models/weights_xception/weights.15.h5', by_name=True)

class_order = ['none', 'screw']

imgs = os.listdir(home + '/ownCloud/imagine_images/screw_data/test1/non_screw/')

tp,tn,fp,fn = 0,0,0,0
for name in imgs:
   image =cv2.imread(home + '/ownCloud/imagine_images/screw_data/test1/non_screw/' + name)
   try:
      image = cv2.resize(image, IMAGE_SIZE)
   except:
      print(name)
      continue
   np_image_data = np.asarray(image)
   np_image_data = np_image_data/255
   np_final = np.expand_dims(np_image_data,axis=0)
   aa = model.predict(np_final)
   if aa[0][0]>0.5:
     tn+=1
   else:
     fn +=1


imgs = os.listdir(home + '/ownCloud/imagine_images/screw_data/test1/screw/)
for name in imgs:
   image =cv2.imread(home + '/ownCloud/imagine_images/screw_data/test1/screw/' + name)  
   try:
       image = cv2.resize(image, IMAGE_SIZE)
   except:
       print(name)
       continue
   np_image_data = np.asarray(image)
   np_image_data = np_image_data/255
   np_final = np.expand_dims(np_image_data,axis=0)
   aa = model.predict(np_final)
   if aa[0][0]>0.5:
     fp+=1
   else:
     tp +=1


accuracy = (tp+tn)/(tp+tn+fp+fn)
print('================================')
print('TP: ', tp, ' TN: ', tn, ' FP: ', fp, ' FN: ', fn)
print('accuracy: ', accuracy)

