from __future__ import print_function, division
import tensorflow as tf
from keras.applications import InceptionV3, Xception
from keras.models import *
from keras.layers import *
import numpy as np
import time, cv2, os
from os.path import expanduser

# to get relative paths without using the user name
home = expanduser('~')

screw_dir = home + '/ownCloud/imagine_images/screw_data/test1/screw/'
nscrew_dir= home + '/ownCloud/imagine_images/screw_data/test1/non_screw/'

IMAGE_SIZE1    = (71, 71)
NUM_CLASSES   = 2
class MModel:    
    @staticmethod
    def loadmodel(path):
        #model = Model(input_img, output_img)
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
        #model = Model(input_img, output_img)
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


screwnames = os.listdir(screw_dir)
nscrewnames = os.listdir(nscrew_dir)

tps,tns,fps,fns, acc=[0]*80,[0]*80,[0]*80,[0]*80,[]
tp,tn,fp,fn=0,0,0,0
for ind, name in enumerate(screwnames):
    imagename = screw_dir+name
    
    img_raw = cv2.imread(imagename)
    image = cv2.resize(img_raw, IMAGE_SIZE1)
    np_image_data = np.asarray(image)
    np_image_data = np_image_data/255
    np_final = np.expand_dims(np_image_data,axis=0)
    aa = model1.predict(np_final)
    #print(aa)

    image2 = cv2.resize(img_raw, IMAGE_SIZE2)
    np_image_data2 = np.asarray(image2)
    np_image_data2 = np_image_data2/255
    np_final2 = np.expand_dims(np_image_data2,axis=0)
    aa2 = model2.predict(np_final2)
    score = aa[0][1]+aa2[0][1]
    thresh = 0.7
    step = 0.01
    for aa in range(79):
      if score>thresh:
        tps[aa] +=1
      else:
        fps[aa] +=1
      thresh+=step
    if score>0.8:
        tp +=1
    else:
        fp +=1

for ind, name in enumerate(nscrewnames):
    imagename = nscrew_dir+name
    
    img_raw = cv2.imread(imagename)
    image = cv2.resize(img_raw, IMAGE_SIZE1)
    np_image_data = np.asarray(image)
    np_image_data = np_image_data/255
    np_final = np.expand_dims(np_image_data,axis=0)
    aa = model1.predict(np_final)

    image2 = cv2.resize(img_raw, IMAGE_SIZE2)
    np_image_data2 = np.asarray(image2)
    np_image_data2 = np_image_data2/255
    np_final2 = np.expand_dims(np_image_data2,axis=0)
    aa2 = model2.predict(np_final2)
    
    score = 0.8*aa[0][1]+1.2*aa2[0][1]
    thresh = 0.7
    step = 0.01
    for aa in range(79):
      if score>thresh:
        fns[aa] +=1
      else:
        tns[aa] +=1
      thresh+=step
    if score>0.8:
        fn +=1
    else:
        tn +=1

for i in range(79):
  try:
    accuracy = (tps[i]+tns[i])/(tps[i]+tns[i]+fps[i]+fns[i])
    acc.append(accuracy)
  except:
    pass
print('maximum accuracy: ',max(acc))

accuracy = (tp+tn)/(tp+tn+fp+fn)
print('TP: ', tp, ' TN: ', tn, ' FP: ', fp, ' FN: ', fn)
print('accuracy: ', accuracy)

