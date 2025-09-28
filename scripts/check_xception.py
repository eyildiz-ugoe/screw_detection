from __future__ import print_function, division
import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import os.path
import keras
from keras import applications, metrics, layers, models, regularizers, optimizers
from keras.applications import InceptionV3, Xception
from keras.models import *
from keras.layers import *
import numpy as np
import cv2
import traceback
from os.path import expanduser

# to get relative paths without using the user name
home = expanduser('~')
base_path = home + "/usr/nishidalab_ws/screw_detection_app/ProjectApplications/scripts/crops/"

IMAGE_SIZE = (71, 71)
NUM_CLASSES = 2

base_model = Xception(include_top=False,
                        weights=None,
                        input_tensor=None,
                        input_shape=(IMAGE_SIZE[0],IMAGE_SIZE[1],3))

x = base_model.output
x = GlobalAveragePooling2D(name='avg_pool')(x)
x = Dropout(0.5)(x)

predictions = Dense(NUM_CLASSES, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

model.load_weights(home + '/usr/nishidalab_ws/screw_detection/models/xception.h5', by_name=True)

class_order = ['none', 'screw']

imgs = os.listdir(base_path)

for name in imgs:
    image = cv2.imread(base_path + name)

    try:
        image = cv2.resize(image, IMAGE_SIZE)
    except Exception as e:
        print(e)
        traceback.print_exc()
        continue

    np_image_data = np.asarray(image)
    np_image_data = np_image_data/255
    np_final = np.expand_dims(np_image_data, axis=0)
    output = model.predict(np_final)

    screw_score = output[0][0]
    # print(screw_score)

    if screw_score > 0.5:
        print(f"{name}: score={screw_score} none screw")
    else:
        print(f"{name}: score={screw_score} screw!!")
