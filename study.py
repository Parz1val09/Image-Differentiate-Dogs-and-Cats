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

#2 - 4
dog2_file = 'data/test/DOG/9375.jpg'
dog3_file = 'data/test/DOG/9376.jpg'

#3 - 6
cat2_file = 'data/test/CAT/9374.jpg'
cat3_file = 'data/test/CAT/9375.jpg'

#4 = 8
cat4_file = 'data/test/CAT/9376.jpg'
cat5_file = 'data/test/CAT/9377.jpg'

#5 - 10
dog4_file = 'data/test/DOG/9377.jpg'
cat6_file = 'data/test/CAT/9378.jpg'

#6 - 12
dog5_file = 'data/test/DOG/9378.jpg'
dog6_file = 'data/test/DOG/9379.jpg'

#7 - 14
cat7_file = 'data/test/CAT/9379.jpg'
dog7_file = 'data/test/DOG/9380.jpg'

#8 - 16
cat8_file = 'data/test/CAT/9380.jpg'
dog8_file = 'data/test/DOG/9381.jpg'

#9 - 18
cat9_file = 'data/test/CAT/9381.jpg'
cat10_file = 'data/test/CAT/9382.jpg'

#10 - 20
dog9_file = 'data/test/DOG/9382.jpg'
cat11_file = 'data/test/CAT/9383.jpg'

####################################################################################
#1
cat_img = image.load_img(cat_file, target_size=(150,150))
cat_img = image.img_to_array(cat_img)

dog_img = image.load_img(dog_file, target_size=(150,150))
dog_img = image.img_to_array(dog_img)

#2
dog2_img = image.load_img(dog2_file, target_size=(150,150))
dog2_img = image.img_to_array(dog2_img)

dog3_img = image.load_img(dog3_file, target_size=(150,150))
dog3_img = image.img_to_array(dog3_img)

#3
cat2_img = image.load_img(cat2_file, target_size=(150,150))
cat2_img = image.img_to_array(cat2_img)

cat3_img = image.load_img(cat3_file, target_size=(150,150))
cat3_img = image.img_to_array(cat3_img)

#4
cat4_img = image.load_img(cat4_file, target_size=(150,150))
cat4_img = image.img_to_array(cat4_img)

cat5_img = image.load_img(cat5_file, target_size=(150,150))
cat5_img = image.img_to_array(cat5_img)

#5
dog4_img = image.load_img(dog4_file, target_size=(150,150))
dog4_img = image.img_to_array(dog4_img)

cat6_img = image.load_img(cat6_file, target_size=(150,150))
cat6_img = image.img_to_array(cat6_img)

#6
dog5_img = image.load_img(dog5_file, target_size=(150,150))
dog5_img = image.img_to_array(dog5_img)

dog6_img = image.load_img(dog6_file, target_size=(150,150))
dog6_img = image.img_to_array(dog6_img)

#7
cat7_img = image.load_img(cat7_file, target_size=(150,150))
cat7_img = image.img_to_array(cat7_img)

dog7_img = image.load_img(dog7_file, target_size=(150,150))
dog7_img = image.img_to_array(dog7_img)

#8
cat8_img = image.load_img(cat8_file, target_size=(150,150))
cat8_img = image.img_to_array(cat8_img)

dog8_img = image.load_img(dog8_file, target_size=(150,150))
dog8_img = image.img_to_array(dog8_img)

#9
cat9_img = image.load_img(cat9_file, target_size=(150,150))
cat10_img = image.img_to_array(cat9_img)

cat10_img = image.load_img(cat10_file, target_size=(150,150))
cat10_img = image.img_to_array(cat10_img)

#10
dog9_img = image.load_img(dog9_file, target_size=(150,150))
dog9_img = image.img_to_array(dog9_img)

cat11_img = image.load_img(cat11_file, target_size=(150,150))
cat11_img = image.img_to_array(cat11_img)

####################################################################################

#1
cat_img = np.expand_dims(cat_img, axis=0)
cat_img = cat_img/255

dog_img = np.expand_dims(dog_img, axis=0)
dog_img = dog_img/255

#2
dog2_img = np.expand_dims(dog2_img, axis=0)
dog2_img = dog2_img/255

dog3_img = np.expand_dims(dog3_img, axis=0)
dog3_img = dog3_img/255

#3
cat2_img = np.expand_dims(cat2_img, axis=0)
cat2_img = cat2_img/255

cat3_img = np.expand_dims(cat3_img, axis=0)
cat3_img = cat3_img/255

#4
cat4_img = np.expand_dims(cat4_img, axis=0)
cat4_img = cat4_img/255

cat5_img = np.expand_dims(cat5_img, axis=0)
cat5_img = cat5_img/255

#5
dog4_img = np.expand_dims(dog4_img, axis=0)
dog4_img = dog4_img/255

cat6_img = np.expand_dims(cat6_img, axis=0)
cat6_img = cat6_img/255

#6
dog5_img = np.expand_dims(dog5_img, axis=0)
dog5_img = dog5_img/255

dog6_img = np.expand_dims(dog6_img, axis=0)
dog6_img = dog6_img/255

#7
cat7_img = np.expand_dims(cat7_img, axis=0)
cat7_img = cat7_img/255

dog7_img = np.expand_dims(dog7_img, axis=0)
dog7_img = dog7_img/255

#8
cat8_img = np.expand_dims(cat8_img, axis=0)
cat8_img = cat8_img/255

dog8_img = np.expand_dims(dog8_img, axis=0)
dog8_img = dog8_img/255

#9
cat9_img = np.expand_dims(cat9_img, axis=0)
cat9_img = cat9_img/255

cat10_img = np.expand_dims(cat10_img, axis=0)
cat10_img = cat10_img/255

#10
dog9_img = np.expand_dims(dog9_img, axis=0)
dog9_img = dog9_img/255

cat11_img = np.expand_dims(cat11_img, axis=0)
cat11_img = cat11_img/255

####################################################################################

#1
a = model.predict(cat_img)

#2
b = model.predict(dog_img)

#3
c = model.predict(dog2_img)

#4
d = model.predict(dog3_img)

#5
e = model.predict(cat2_img)

#6
f = model.predict(cat3_img)

#7
g = model.predict(cat4_img)

#8
h = model.predict(cat5_img)

#9
i = model.predict(dog4_img)

#10
j = model.predict(cat6_img)

#11
k = model.predict(dog5_img)

#12
l = model.predict(dog6_img)

#13
m = model.predict(cat7_img)

#14
n = model.predict(dog7_img)

#15
o = model.predict(cat8_img)

#16
p = model.predict(dog8_img)

#17
q = model.predict(cat9_img)

#18
r = model.predict(cat10_img)

#19
s = model.predict(dog9_img)

#20
t = model.predict(cat11_img)

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

#3
if c > 0.5:
    print('Dog == Dog')

else:
    print('Cat != Dog')

#4
if d > 0.5:
    print('Dog == Dog')

else:
    print('Cat != Dog')

#5
if e > 0.5:
    print('Cat != Dog')

else:
    print('Cat == Cat')

#6
if f > 0.5:
    print('Cat != Dog')

else:
    print('Cat == Cat')

#7
if g > 0.5:
    print('Cat != Dog')

else:
    print('Cat == Cat')

#8
if h > 0.5:
    print('Cat != Dog')

else:
    print('Cat == Cat')

#9
if i > 0.5:
    print('Dog == Dog')

else:
    print('Cat != Dog')

#10
if j > 0.5:
    print('Cat != Dog')

else:
    print('Cat == Cat')

#11
if k > 0.5:
    print('Dog == Dog')

else:
    print('Cat != Dog')

#12
if l > 0.5:
    print('Dog == Dog')

else:
    print('Cat != Dog')

#13
if m > 0.5:
    print('Cat != Dog')

else:
    print('Cat == Cat')

#14
if n > 0.5:
    print('Dog == Dog')

else:
    print('Cat != Dog')

#15
if o > 0.5:
    print('Cat != Dog')

else:
    print('Cat == Cat')

#16
if p > 0.5:
    print('Dog == Dog')

else:
    print('Cat != Dog')

#17
if q > 0.5:
    print('Cat != Dog')

else:
    print('Cat == Cat')

#18
if r > 0.5:
    print('Cat != Dog')

else:
    print('Cat == Cat')

#19
if s > 0.5:
    print('Dog == Dog')

else:
    print('Cat != Dog')

#20
if t > 0.5:
    print('Cat != Dog')

else:
    print('Cat == Cat')

