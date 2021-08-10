import numpy as np
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.mobilenet import preprocess_input

train_size=2183
test_size=501
batch_size=16

#test data, no transformation
datagen_validation = ImageDataGenerator(preprocessing_function=preprocess_input)

valid_generator = datagen_validation.flow_from_directory(directory="data/test/", target_size=(100, 100), color_mode="rgb", batch_size=1, class_mode="categorical", shuffle=False, seed=42)

#get models for comparison
models = ['google2','google3','google4','google5','V3']

for model in models:
	file_dir = 'models/' + model
	model = load_model(file_dir)

	label_map = {0: 'eagle', 1: 'vulture'}

	#Confution Matrix and Classification Report
	Y_pred = model.predict(valid_generator, test_size // batch_size+1)
	y_pred = np.argmax(Y_pred, axis=1)
	
	#print model name
	print(file_dir)

	print('Confusion Matrix')
	print(confusion_matrix(valid_generator.classes, y_pred))
	print('Classification Report')
	target_names = ['eagle', 'vulture']
	print(classification_report(valid_generator.classes, y_pred, target_names=target_names))
 
 