import sqlite3
from flask import Flask, render_template, request
import numpy as np
import os
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import load_model

# Connect to database
conn = sqlite3.connect('example.db')
cursor = conn.cursor()

# Execute a SQL query to create the table if it doesn't exist
cursor.execute("CREATE TABLE IF NOT EXISTS my_table (column1 TEXT, column2 TEXT)")

# Close the cursor and commit the changes to the database
cursor.close()
conn.commit()
conn.close()

# Define the plant types and their corresponding models
plants = {
    'Tomato': {
        'model_path': 'E:/Plant_Leaf_Disease_Prediction/Tomato_Leaf_Diseases_Prediction/InceptionV3_256.h5',
        'threshold': 0.95,
        'classes': {
            0: ("Tomato - Bacteria Spot Disease", 'Tomato-Bacteria Spot.html'),
            1: ("Tomato - Early Blight Disease", 'Tomato-Early_Blight.html'),
            2: ("Tomato - Late Blight Disease", 'Tomato - Late_blight.html'),
            3: ("Tomato - Leaf Mold Disease", 'Tomato - Leaf_Mold.html'),
            4: ("Tomato - Septoria Leaf Spot Disease", 'Tomato - Septoria_leaf_spot.html'),
            5: ("Tomato - Two Spotted Spider Mite Disease", 'Tomato - Two-spotted_spider_mite.html'),
            6: ("Tomato - Target Spot Disease", 'Tomato - Target_Spot.html'),
            7: ("Tomato - Tomato Yellow Leaf Curl Virus Disease", 'Tomato - Tomato_Yellow_Leaf_Curl_Virus.html'),
            8: ("Tomato - Tomato Mosaic Virus Disease", 'Tomato - Tomato_mosaic_virus.html'),
            9: ("Tomato - Healthy and Fresh", 'Tomato-Healthy.html')
        }
    },
    'Corn': {
        'model_path': 'E:/Plant_Leaf_Disease_Prediction/Corn_Leaf_Diseases_Prediction/model.h5',
        'threshold': 0.8,
        'classes': {
            0: ("Corn - Blight Disease", 'Corn - Blight.html'),
            1: ("Corn - Common Rust Disease", 'Corn - Common_Rust.html'),
            2: ("Corn - Gray Leaf Spot Disease", 'Corn - Gray_Leaf_Spot.html'),
            3: ("Corn - Healthy and Fresh", 'Corn - Healthy.html')
        }
    },
    'Potato': {
        'model_path': 'E:/Plant_Leaf_Disease_Prediction/Potato_Leaf_Diseases_Prediction/model.h5',
        'threshold': 0.95,
        'classes': {
            0: ("Potato - Early Blight Disease", 'Potato_Early_Blight.html'),
            1: ("Potato - Healthy", 'Potato_Healthy.html'),
            2: ("Potato - Late Blight", 'Potato_Late_Blight.html')
        }
    },
    'Mango': {
        'model_path': 'E:/Plant_Leaf_Disease_Prediction/Mango_Leaf_Diseases_Prediction/model.h5',
        'threshold': 0.95,
        'classes': {
            0: ("Mango - Anthracnose Disease", 'Mango - Anthracnose.html'),
            1: ("Mango - Bacterial Canker Disease", 'Mango - Bacterial_Canker.html'),
            2: ("Mango - Cutting Weevil Disease", 'Mango - Cutting_Weevil.html'),
            3: ("Mango - Die Back Disease", 'Mango - Die_Back.html'),
            4: ("Mango - Gall Midge Disease", 'Mango - Gall_Midge.html'),
            5: ("Mango - Healthy", 'Mango - Healthy.html'),
            6: ("Mango - Powdery Mildew Disease", 'Mango - Powdery_Mildew.html'),
            7: ("Mango - Sooty Mould Disease", 'Mango - Sooty_Mould.html')
        }
    },
    'Pepper': {
        'model_path': 'E:/Plant_Leaf_Disease_Prediction/Pepper_Leaf_Diseases_Prediction/model.h5',
        'threshold': 0.05,
        'classes': {
            0: ("Pepper Bell Bacterial Spot", 'Pepper_Bell_Bacterial_spot.html'),
            1: ("Pepper Bell Healthy", 'Pepper_Bell_Healthy.html'),
            }
        },
    'Apple': {
        'model_path': 'E:/Plant_Leaf_Disease_Prediction/Apple_Leaf_Diseases_Prediction/model.h5',
        'threshold': 0.95,
        'classes': {
            0: ("Apple - Black Rot", 'Apple - Black_Rot.html'),
            1: ("Apple - Healthy", 'Apple - Healthy.html'),
            2: ("Apple - Scab", 'Apple - Scab.html'),
            3: ("Apple - Cedar Apple Rust", 'Apple - Cedar_Apple_Rust.html')
        }
    },
    'Strawberry': {
        'model_path': 'E:/Plant_Leaf_Disease_Prediction/Strawberry_Leaf_Diseases_Prediction/model.h5',
        'threshold': 0.95,
        'classes': {
            0: ("Strawberry - Healthy", 'Strawberry - Healthy.html'),
            1: ("Strawberry - Leaf Scorch Disease", 'Strawberry - Leaf_Scorch.html')
        }
    },
    'Rice': {
        'model_path': 'E:/Plant_Leaf_Disease_Prediction/Rice_Leaf_Diseases_Prediction/model.h5',
        'threshold': 0.8,
        'classes': {
            0: ("Rice - Brown Spot", 'Rice - BrownSpot.html'),
            1: ("Rice - Healthy", 'Rice - Healthy.html'),
            2: ("Rice - Hispa", 'Rice - Hispa.html'),
            3: ("Rice - Leaf Blast", 'Rice - LeafBlast.html')
        }
    }
    
}

# Load the models lazily when needed
models = {}

def load_models():
    for plant, config in plants.items():
        model_path = config['model_path']
        model = load_model(model_path)
        models[plant] = model
        print(f"Model for {plant} loaded successfully")

def get_model(plant_type):
    if plant_type in models:
        return models[plant_type]
    else:
        return None
    
def classify_image(image_path):
    # Load the classification model from C:/Desktop
    classification_model_path = 'C:/Desktop/model.h5'
    classification_model = load_model(classification_model_path)
    print("Classification model loaded successfully")

    # Load the image and resize it
    target_size = (128, 128)
    try:
        test_image = load_img(image_path, target_size=target_size)
        print("@@ Got Image for classification")
    except Exception as e:
        print("@@ Error loading image:", str(e))
        return "Unknown"

    # Convert the image to a numpy array and normalize it
    test_image = img_to_array(test_image) / 255
    test_image = np.expand_dims(test_image, axis=0)

    # Make predictions using the classification model
    result = classification_model.predict(test_image)
    print('@@ Classification result:', result)

    # Get the predicted class
    pred_class = np.argmax(result, axis=1)[0]  # Convert to scalar value

    # Define the class labels for classification
    class_labels = {
        0: 'Cat',
        1: 'Dog',
        2: 'Human',
        3: 'Leaf',
        4: 'Panda'
    }

    if pred_class in class_labels:
        return class_labels[pred_class]
    else:
        return "Unknown"

def classify_image(image_path):
    model_path = 'E:/Plant_Leaf_Disease_Prediction/Leaf_Detection/model.h5'
    model = load_model(model_path)
    print("Classification model loaded successfully")
    target_size = (128, 128)  # Adjust the target size as needed

    try:
        test_image = load_img(image_path, target_size=target_size)
        print("@@ Got Image for classification")
    except Exception as e:
        print("@@ Error loading image:", str(e))
        return "Unknown"

    test_image = img_to_array(test_image) / 255
    test_image = np.expand_dims(test_image, axis=0)

    classes = ['Cat', 'Dog', 'Human', 'Leaf', 'Panda']  # Add more classes if needed

    result = model.predict(test_image)
    pred_class = np.argmax(result, axis=1)[0]
    pred_label = classes[pred_class]

    return pred_label

def pred_disease(plant_type, image_path):
    model = get_model(plant_type)
    if model is None:
        return "Unknown Disease", 'unknown.html'

    # Load the image and resize it
    target_size = (128, 128)  # Default target size for Tomato, can be adjusted for other plant types
    if plant_type == 'Tomato':
        target_size = (256, 256)
    
    try:
        test_image = load_img(image_path, target_size=target_size)
        print("@@ Got Image for prediction")
    except Exception as e:
        print("@@ Error loading image:", str(e))
        return "Error loading image", 'error.html'

    # Convert the image to a numpy array and normalize it
    test_image = img_to_array(test_image) / 255
    test_image = np.expand_dims(test_image, axis=0)

    # Make predictions using the model for the selected plant type
    result = model.predict(test_image)
    print('@@ Raw result =', result)

    # Get the predicted class and its confidence
    pred_class = np.argmax(result, axis=1)[0]  # Convert to scalar value
    pred_confidence = np.max(result)

    threshold = plants[plant_type].get('threshold', 0.95)
    if pred_confidence < threshold:
        return "Unknown Disease", 'unknown.html'

    # Get the class information for the selected plant type
    classes = plants[plant_type]['classes']
    if pred_class in classes:
        return classes[pred_class]
    else:
        return "Unknown Disease", 'unknown.html'


# Create Flask instance
app = Flask(__name__)

# Render index.html page
@app.route("/", methods=['GET', 'POST'])
def home():
    return render_template('index.html')

# Get input plant type and image from client, predict class, and render respective .html page for solution
@app.route("/predict", methods=['POST'])
@app.route("/predict", methods=['POST'])
def predict():
    plant_type = request.form.get('plant_type')
    file = request.files.get('image')

    if not file:
        return "Error: No file uploaded"

    filename = file.filename
    print("@@ Input posted =", filename)

    file_path = os.path.join('static/upload', filename)
    file.save(file_path)

    print("@@ Classifying image...")
    image_class = classify_image(file_path)

    if image_class == 'Leaf':
        print("@@ Predicting disease...")
        pred, output_page = pred_disease(plant_type, image_path=file_path)
        return render_template(output_page, pred_output=pred, user_image=file_path)
    else:
        return render_template('unknown.html')
# Load models before running the application
load_models()

# For local system & cloud
if __name__ == "__main__":
    app.run(threaded=False, port=2020)
