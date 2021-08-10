import os
import tensorflow as tf
from flask import Flask, request, redirect, url_for, send_from_directory, render_template
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import load_model
from werkzeug.utils import secure_filename
import numpy as np

ALLOWED_EXTENSIONS = set(['jpg', 'jpeg', 'png'])
IMAGE_SIZE = (100, 100)
UPLOAD_FOLDER = 'uploads'

google = load_model('model/google')

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS

def predict(img_path):

  label_map = {0: 'eagle', 1: 'vulture'}

  img = tf.keras.preprocessing.image.load_img(img_path,target_size=(100,100) )
  img = tf.keras.preprocessing.image.img_to_array(img)
  img = img.reshape((1,) + img.shape)
  #img.shape

  pred = google.predict(img)[0][0]
  
  if label_map[pred] == 'eagle':
      return 'That looks like an eagle to me!'
  else:
      return 'That looks like a vulture to me!'

#  return label_map[pred]

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route("/")
def template_test():
    return render_template('home.html', label='', imagesource='file://null')

@app.route('/home')
def home():
    return render_template('home.html', label='', imagesource='file://null')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/eagle')
def eagle():
    return render_template('eagle.html')

@app.route('/owl')
def owl():
    return render_template('owl.html')

@app.route('/ref')
def ref():
    return render_template('ref.html')

@app.route('/vulture')
def vulture():
    return render_template('vulture.html')

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['file']
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            output = predict(file_path)
    return render_template("home.html", label=output, imagesource=file_path)

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'],
                               filename)

if __name__ == "__main__":
    app.run(debug=False, threaded=False)