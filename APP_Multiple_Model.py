import sqlite3
from flask import Flask, render_template, request
import numpy as np
import os
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import load_model

# Connect to database
conn = sqlite3.connect('example.db')

# Create a cursor object to interact with the database
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
        'model_path': 'E:/Tomato_Leaf_Diseases_Prediction/InceptionV3_256.h5',
        'classes': {
            0: ("Tomato - Bacteria Spot Disease", 'Tomato-Bacteria Spot.html'),
            1: ("Tomato - Early Blight Disease", 'Tomato-Early_Blight.html'),
            2: ("Tomato - Late Blight Disease", 'Tomato - Late_blight.html'
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
        'classes': {
            0: ("Corn - Blight Disease", 'Corn-Bacteria Spot.html'),
            1: ("Corn - Common Rust Disease", 'Corn-Early_Blight.html'),
            2: ("Corn - Gray Leaf Spot Disease", 'Corn-Late_blight.html'),
            3: ("Corn - Healthy and Fresh", 'Corn-Leaf_Mold.html')
        }
    },
    'Maize': {
        'model_path': 'E:/Plant_Leaf_Disease_Prediction/Corn_Leaf_Diseases_Prediction/model.h5',
        'classes': {
            0: ("Maize - Bacteria Spot Disease", 'Maize-Bacteria Spot.html'),
            1: ("Maize - Early Blight Disease", 'Maize-Early_Blight.html'),
            2: ("Maize - Late Blight Disease", 'Maize-Late_blight.html'),
            3: ("Maize - Leaf Mold Disease", 'Maize-Leaf_Mold.html')
        }
    },
    'Mango': {
        'model_path': 'E:/Mango_Leaf_Diseases_Prediction/model3.h5',
        'classes': {
            0: ("Mango - Bacteria Spot Disease", 'Mango-Bacteria Spot.html'),
            1: ("Mango - Early Blight Disease", 'Mango-Early_Blight.html'),
            2: ("Mango - Late Blight Disease", 'Mango-Late_blight.html'),
            3: ("Mango - Leaf Mold Disease", 'Mango-Leaf_Mold.html')
        }
    },
    'Rice': {
        'model_path': 'E:/Rice_Leaf_Diseases_Prediction/model4.h5',
        'classes': {
            0: ("Rice - Bacteria Spot Disease", 'Rice-Bacteria Spot.html'),
            1: ("Rice - Early Blight Disease", 'Rice-Early_Blight.html'),
            2: ("Rice - Late Blight Disease", 'Rice-Late_blight.html'),
            3: ("Rice - Leaf Mold Disease", 'Rice-Leaf_Mold.html')
        }
    }
}

# Load the models
models = {}
for plant, config in plants.items():
    model_path = config['model_path']
    model = load_model(model_path)
    models[plant] = model
    print(f"Model for {plant} loaded successfully")

def pred_disease(plant_type, image_path):
    # Load the image and resize it
    test_image = load_img(image_path, target_size=(256, 256))
    print("@@ Got Image for prediction")

    # Convert the image to a numpy array and normalize it
    test_image = img_to_array(test_image) / 255
    test_image = np.expand_dims(test_image, axis=0)

    # Make predictions using the model for the selected plant type
    model = models[plant_type]
    result = model.predict(test_image)
    print('@@ Raw result = ', result)

    # Get the predicted class
    pred = np.argmax(result, axis=1)
    print(pred)

    # Get the class information for the selected plant type
    classes = plants[plant_type]['classes']
    if pred in classes:
        return classes[pred]
    else:
        return "Unknown Disease", 'unknown.html'

# Create Flask instance
app = Flask(__name__)

# Render index.html page
@app.route("/", methods=['GET', 'POST'])
def home():
    return render_template('index.html')

# Get input plant type and image from client, predict class, and render respective .html page for solution
@app.route("/predict", methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        plant_type = request.form.get('plant_type')
        file = request.files['image']
        filename = file.filename
        print("@@ Input posted = ", filename)

        file_path = os.path.join('E:/Leaf_Diseases_Prediction/App/static/upload', filename)
        file.save(file_path)

        print("@@ Predicting class...")
        pred, output_page = pred_disease(plant_type, image_path=file_path)

        return render_template(output_page, pred_output=pred, user_image=file_path)

# For local system & cloud
if __name__ == "__main__":
    app.run(threaded=False, port=4040)
