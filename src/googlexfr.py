# import the necessary packages
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import concatenate
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
from minigooglenet import minigooglenet_functional

#set param values
#classes (eagles, vultures)
n_categories = 2
dir_train = '.'
train_size = 2183
test_size = 501
batch_size = 16
EPOCHS = 6

#train data - 2 classes, 1000 per class
datagen_train = ImageDataGenerator(preprocessing_function=preprocess_input,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')

#test data, no transformation
datagen_validation = ImageDataGenerator(preprocessing_function=preprocess_input)

#load images while model is running
train_generator = datagen_train.flow_from_directory(
   directory='./data/train/', 
   target_size=(100,100),
   color_mode='rgb',
   batch_size=32,
   class_mode='categorical',
   shuffle=True,
   seed=42)

valid_generator = datagen_validation.flow_from_directory(
    directory="./data/test/",
    target_size=(100, 100),
    color_mode="rgb",
    batch_size=1,
    class_mode="categorical",
    shuffle=False,
    seed=42)

#create model
google = minigooglenet_functional(100, 100, 3, n_categories)

#compile model with very slow learning rate
google.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5), loss='categorical_crossentropy', metrics=['accuracy'])

history = google.fit(train_generator, steps_per_epoch=train_size//batch_size, epochs=1, validation_data=valid_generator, validation_steps= test_size//batch_size)

#save model
google.save('models/googlexfr')

#analyze results

#Confution Matrix and Classification Report
Y_pred = google.predict(valid_generator, test_size // batch_size+1)
y_pred = np.argmax(Y_pred, axis=1)

print('Confusion Matrix')
print(confusion_matrix(valid_generator.classes, y_pred))
print('Classification Report')
target_names = ['eagle', 'vulture']
print(classification_report(valid_generator.classes, y_pred, target_names=target_names))



