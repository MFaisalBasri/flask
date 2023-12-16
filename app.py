from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import numpy as np
from google.cloud import storage
import io

app = Flask(__name__)

# Load the pre-trained model
model = load_model("makara.h5")


def preprocess_image(file):
    # Load image and resize
    img = Image.open(file).resize((224, 224))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    # Check if a file was uploaded
    if 'file' not in request.files:
        return render_template('index.html', message='No file part')

    file = request.files['file']

    # Check if the file is empty
    if file.filename == '':
        return render_template('index.html', message='No selected file')

    # Check if the file is allowed
    allowed_extensions = {'png', 'jpg', 'jpeg', 'gif'}
    if '.' not in file.filename or file.filename.split('.')[-1].lower() not in allowed_extensions:
        return render_template('index.html', message='Invalid file type')

    # Preprocess the image
    img_array = preprocess_image(file)

    # Make prediction
    classes = model.predict(img_array, batch_size=8)

    # You may want to process the prediction result and display it on the page
    # For now, let's assume you want to display the class with the highest probability
    class_index = np.argmax(classes)
    # class_label = f"Class {class_index}"

    if np.argmax(classes) == 0:
        class_label = f"Bika Ambon"
    elif np.argmax(classes) == 1:
        class_label = f"Kerak Telor"
    elif np.argmax(classes) == 2:
        class_label = f"Molen"
    elif np.argmax(classes) == 3:
        class_label = f"Nasi Goreng"
    elif np.argmax(classes) == 4:
        class_label = f"Papeda Maluku"
    elif np.argmax(classes) == 5:
        class_label = f"Sate Padang"
    elif np.argmax(classes) == 6:
        class_label = f"Seblak"
    else:
        print("Class not recognized")

    return render_template('index.html', message='Prediction:', prediction=class_label)


if __name__ == '__main__':
    app.run(debug=True)
