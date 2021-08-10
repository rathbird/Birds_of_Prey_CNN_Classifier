import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow.keras.layers import Dense, GlobalAveragePooling2D 
from tensorflow.keras.models import Model

from tensorflow.keras.applications.xception import preprocess_input

from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from tensorflow.keras.optimizers import RMSprop

from utility import change_trainable_layers

#set param values
#classes (eagles, vultures)
n_categories = 2
batch_size = 16
dir_train = '.'
train_size = 2183
test_size = 501
EPOCHS = 20


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


# Initialize a pretrained model with the Xception architecture and weights pretrained on imagenet
input_size = (100,100,3)
base_model = tf.keras.applications.Xception(weights='imagenet',
			include_top=False,
			input_shape=input_size)

model = base_model.output
model = GlobalAveragePooling2D()(model)
predictions = Dense(n_categories, activation='softmax')(model)
model = Model(inputs=base_model.input, outputs=predictions)

#change trainable layers
_ = change_trainable_layers(model, 132)

# Compile model
model.compile(optimizer=RMSprop(lr=0.0005), loss='categorical_crossentropy', metrics=['accuracy'])

# Train model for 10-30 epochs on AWS
model.fit(train_generator, epochs=EPOCHS, validation_data=valid_generator)

#save model
model.save('models/xception')

#analyze model accuracy

label_map = {0: 'eagle', 1: 'vulture'}

from sklearn.metrics import classification_report, confusion_matrix
#Confution Matrix and Classification Report
Y_pred = model.predict(valid_generator, test_size // batch_size+1)
y_pred = np.argmax(Y_pred, axis=1)

print('Confusion Matrix')
print(confusion_matrix(valid_generator.classes, y_pred))
print('Classification Report')
target_names = ['eagle', 'vulture']
print(classification_report(valid_generator.classes, y_pred, target_names=target_names))






