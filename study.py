import matplotlib.pyplot as plt
import cv2
import numpy as np
import keras.utils as image

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Activation, Dropout, Flatten, Conv2D, MaxPooling2D, Dense
from keras.optimizers import Adam

import tensorflow as tf

# Evitar erros OOM por consumo em acesso da memória da GPU
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus: 
    tf.config.experimental.set_memory_growth(gpu, True)

# Load and preprocess example images
cat4 = cv2.imread('data/train/CAT/4.jpg')
cat4 = cv2.cvtColor(cat4, cv2.COLOR_BGR2RGB)
cat4 = cv2.resize(cat4, (150, 150))
cat4 = cat4.astype('float32') / 255.0

dog = cv2.imread('data/train/DOG/2.jpg')
dog = cv2.cvtColor(dog, cv2.COLOR_BGR2RGB)
dog = cv2.resize(dog, (150, 150))
dog = dog.astype('float32') / 255.0

# Define image augmentation generator
image_gen = ImageDataGenerator(rotation_range=30,
                               width_shift_range=0.1,
                               height_shift_range=0.1,
                               rescale=1/255,
                               shear_range=0.2,
                               zoom_range=0.2,
                               horizontal_flip=True,
                               fill_mode='nearest')

# Define input shape and create model
input_shape = (150,150,3)

model = Sequential()

model.add(Conv2D(filters=32, kernel_size=(3,3), input_shape=(150,150,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(filters=64, kernel_size=(3,3), input_shape=(150,150,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(filters=64, kernel_size=(3,3), input_shape=(150,150,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())

model.add(Dense(128))
model.add(Activation('relu'))

model.add(Dropout(0.5))

model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.001), metrics=['acc'])
model.summary()

# Train model
batch_size = 16
train_image_gen = image_gen.flow_from_directory('data/train', 
                                                target_size=input_shape[:2],
                                                batch_size=batch_size,
                                                class_mode='binary')

test_image_gen = image_gen.flow_from_directory('data/test', 
                                                target_size=input_shape[:2],
                                                batch_size=batch_size,
                                                class_mode='binary')

epochs = 50
steps_per_epoch = int(np.ceil(train_image_gen.samples / batch_size))

validation_steps = int(np.ceil(test_image_gen.samples / batch_size))

results = model.fit_generator(train_image_gen, epochs=epochs, steps_per_epoch=steps_per_epoch, validation_steps=validation_steps)

import warnings
warnings.filterwarnings('ignore')

print(train_image_gen.class_indices)
print(results.history['acc'])

from keras.models import load_model
import os
model.save(os.path.join('models','cat_dog_100epochs.h5'))
new_model = load_model(os.path.join('models', 'cat_dog_100epochs.h5'))
#1 - 2
cat_file = 'data/test/CAT/9584.jpg'
dog_file = 'data/test/DOG/9374.jpg'

####################################################################################
#1
cat_img = image.load_img(cat_file, target_size=(150,150))
cat_img = image.img_to_array(cat_img)

dog_img = image.load_img(dog_file, target_size=(150,150))
dog_img = image.img_to_array(dog_img)

####################################################################################

#1
cat_img = np.expand_dims(cat_img, axis=0)
cat_img = cat_img/255

dog_img = np.expand_dims(dog_img, axis=0)
dog_img = dog_img/255

####################################################################################

#1
a = model.predict(cat_img)

#2
b = model.predict(dog_img)

#1
if a > 0.5:
    print('Cat != Dog')

else:
    print('Cat == Cat')

#2
if b > 0.5:
    print('Dog == Dog')

else:
    print('Cat != Dog')
