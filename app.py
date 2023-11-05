import os
import cv2
import numpy as np
from PIL import Image
from flask import Flask, render_template, request
import tensorflow as tf

app = Flask(__name__)

# Load the trained model
model_path = 'my_model.h5'  # Replace with the path to your trained model
loaded_model = tf.keras.models.load_model(model_path)

# Function to preprocess the uploaded image
def preprocess_image(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # image = cv2.resize(image, (64, 64))
    image = cv2.resize(image, (64, 64), interpolation=cv2.INTER_LINEAR)
    image = np.array(image)
    image = image.astype('float32') / 255.0
    image = np.expand_dims(image, axis=0)
    return image

# Function to predict brain tumor from the uploaded image
def predict_tumor(image):
    preprocessed_image = preprocess_image(image)
    prediction = loaded_model.predict(preprocessed_image)
    if prediction > 0.5:
        label = 'Brain Tumor'
    else:
        label = 'Non-Tumor'
    return label

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Get the uploaded file from the request
        file = request.files['file']

        # Check if the file exists and has an allowed extension
        if file and file.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            # Read and preprocess the uploaded image
            image = Image.open(file)
            # image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            label = predict_tumor(image)

            # Render the result template with the prediction
            return render_template('index.html', prediction=label, filename=file.filename)

    # Render the index template
    return render_template('index.html')

if __name__ == '__main__':
    app.static_folder = 'static'
    app.run(debug=True)
