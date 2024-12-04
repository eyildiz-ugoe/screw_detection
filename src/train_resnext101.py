from __future__ import print_function

import os.path

import keras
from keras import applications, metrics, layers, models, regularizers, optimizers
from keras.applications import DenseNet201, Xception
from resnet import ResNeXt101
from keras.models import *
from keras.layers import *
from keras.callbacks import *
from keras.optimizers import Adam, RMSprop
from keras.preprocessing.image import ImageDataGenerator
from keras.metrics import top_k_categorical_accuracy
from os.path import expanduser

# to get relative paths without using the user name
home = expanduser('~')

DATASET_PATH_train  = home + '/ownCloud/imagine_images/screw_data/train1'
DATASET_PATH_val  = home + '/ownCloud/imagine_images/screw_data/val'

IMAGE_SIZE    = (64, 64)
NUM_CLASSES   = 2
BATCH_SIZE    = 8  # try reducing batch size or freeze more layers if your GPU runs out of memory
FREEZE_LAYERS = 0  # freeze the first this many layers for training
NUM_EPOCHS    = 15

tensorboard_directory   = 'log_files/logs_ResNeXt101'
weight_directory        = home + '/ownCloud/imagine_weights/screw_detector/pretrained_models/weights_ResNeXt101'
WEIGHTS_FINAL = weight_directory+'/model-final.h5'

train_datagen = ImageDataGenerator(rescale=1./255,
                                   rotation_range=20,
                                   width_shift_range=0.1, height_shift_range=0.1, 
                                   shear_range=0.01, zoom_range=[0.9,1.25],
                                   horizontal_flip=True,
				   vertical_flip=True,
				   brightness_range=[0.4,1.5],
                                   fill_mode='reflect')
train_batches = train_datagen.flow_from_directory(DATASET_PATH_train,
                                                  target_size=IMAGE_SIZE,
                                                  class_mode='categorical',
                                                  shuffle=True,
                                                  batch_size=BATCH_SIZE)

valid_datagen = ImageDataGenerator(rescale=1./255)
valid_batches = valid_datagen.flow_from_directory(DATASET_PATH_val,
                                                  target_size=IMAGE_SIZE,
                                                  class_mode='categorical',
                                                  shuffle=False,
                                                  batch_size=BATCH_SIZE)

# show class indices
print('****************')
classes_name = [0 for i in range(NUM_CLASSES)]
for cls, idx in train_batches.class_indices.items():
    print('Class #{} = {}'.format(idx, cls))
    classes_name[idx] = cls
print(classes_name)
print('****************')

# build our classifier model based on pre-trained model:
# 1. we don't include the top (fully connected) layers of pretrained models
# 2. we add a DropOut layer followed by a Dense (fully connected)
#    layer which generates softmax class score for each class
# 3. we compile the final model using an Adam optimizer, with a
#    low learning rate (since we are 'fine-tuning')
base_model = ResNeXt101(include_top=False,
                        weights='imagenet',
                        input_tensor=None,
                        input_shape=(IMAGE_SIZE[0],IMAGE_SIZE[1],3))
for layer in base_model.layers:
    layer.trainable = True

# add a global spatial average pooling layer
x = base_model.output
x = GlobalAveragePooling2D(name='avg_pool')(x)
x = Dropout(0.5)(x)

# and a logistic layer -- let's say we have 2 classes
predictions = Dense(NUM_CLASSES, activation='softmax')(x)

# this is the model we will train
model = Model(inputs=base_model.input, outputs=predictions)

print(len(model.layers))
tensorboard_callback = TensorBoard(log_dir=tensorboard_directory, 
                                                       histogram_freq=0,
                                                       write_graph=True,
                                                       write_images=False)
save_model_callback = ModelCheckpoint(os.path.join(weight_directory, 'weights.{epoch:02d}.h5'),
                                                        verbose=3,
                                                        save_best_only=False,
                                                        save_weights_only=False,
                                                        mode='auto',
                                                        period=1)

#compile model
model.compile(optimizer=Adam(lr=1e-5, beta_1=0.9, beta_2=0.999, epsilon=1e-8, decay=1e-9, amsgrad=True), loss='categorical_crossentropy',
                  metrics=['accuracy'])

print(model.summary())

# train the model
model.fit_generator(train_batches,
                        steps_per_epoch = train_batches.samples // BATCH_SIZE,
                        validation_data = valid_batches, 
                        validation_steps = valid_batches.samples // BATCH_SIZE,
                        callbacks=[save_model_callback, tensorboard_callback],
                        epochs = NUM_EPOCHS)

# save trained weight
model.save(WEIGHTS_FINAL)

