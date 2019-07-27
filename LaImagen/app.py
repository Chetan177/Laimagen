# Dependencies
from flask import Flask,request,render_template,jsonify,json
import base64
import io
from PIL import Image
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array
from tensorflow.keras.models import Sequential, load_model
import matplotlib.pyplot as plt
import cv2
import numpy as np
from load import*
global model
classes = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']
app = Flask(__name__)

def pre_val(res):
  x = np.where(res == np.amax(res))
  x=x[0]
  return x[0]

def preprocess_img(image, target_size):
    image = image.convert("L")
    image = image.resize(target_size)
    image = img_to_array(image)
    image = image.reshape(1,32,32,3)
    return image
@app.route('/')
def welcome():
    return render_template("index.html")

@app.route("/predict", methods = ["GET","POST"])
def predict():
  if request.method == 'POST':
     f = request.files['img']
     in_memory_file = io.BytesIO()
     f.save(in_memory_file)
     data = np.fromstring(in_memory_file.getvalue(), dtype=np.uint8)
     color_image_flag = 1
     img = cv2.imdecode(data, color_image_flag)
     print(type(img))
     print(img.size)
     img = cv2.resize(img,(32, 32))
     img= img.reshape(1,32,32,3)
     pred = model.predict(img)
     idx = pre_val(pred[0])
     res = "The Image is of a "+str(classes[idx])
     return render_template("index.html",output=res)
@app.errorhandler(500)
def internal_error(error):

    return render_template("index.html",output = "Error occur plz reload to main page ")

if __name__ == "__main__":
    model= init()
    app.run(port = 5050) # app.run(host="0.0.0.0")
