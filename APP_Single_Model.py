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

# Path to the pre-trained model
filepath = 'E:/Plant_Leaf_Disease_Prediction/Tomato_Leaf_Diseases_Prediction/InceptionV3_256.h5'

# Load the model
model = load_model(filepath)
print(model)

print("Model Loaded Successfully")

def pred_tomato_disease(tomato_plant):
    # Load the image and resize it
    test_image = load_img(tomato_plant, target_size=(256, 256))
    print("@@ Got Image for prediction")

    # Convert the image to a numpy array and normalize it
    test_image = img_to_array(test_image) / 255
    test_image = np.expand_dims(test_image, axis=0)

    # Make predictions using the model
    result = model.predict(test_image)
    print('@@ Raw result = ', result)

    # Get the predicted class
    pred = np.argmax(result, axis=1)
    print(pred)

    if pred == 0:
        return "Tomato - Bacteria Spot Disease", 'Tomato-Bacteria Spot.html'
    elif pred == 1:
        return "Tomato - Early Blight Disease", 'Tomato-Early_Blight.html'
    elif pred == 2:
        return "Tomato - Late Blight Disease", 'Tomato - Late_blight.html'
    elif pred == 3:
        return "Tomato - Leaf Mold Disease", 'Tomato - Leaf_Mold.html'
    elif pred == 4:
        return "Tomato - Septoria Leaf Spot Disease", 'Tomato - Septoria_leaf_spot.html'
    elif pred == 5:
        return "Tomato - Two Spotted Spider Mite Disease", 'Tomato - Two-spotted_spider_mite.html'
    elif pred == 6:
        return "Tomato - Target Spot Disease", 'Tomato - Target_Spot.html'
    elif pred == 7:
        return "Tomato - Tomato Yellow Leaf Curl Virus Disease", 'Tomato - Tomato_Yellow_Leaf_Curl_Virus.html'
    elif pred == 8:
        return "Tomato - Tomato Mosaic Virus Disease", 'Tomato - Tomato_mosaic_virus.html'
    elif pred == 9:
        return "Tomato - Healthy and Fresh", 'Tomato-Healthy.html'


# Create Flask instance
app = Flask(__name__)

# Render index.html page
@app.route("/", methods=['GET', 'POST'])
def home():
    return render_template('index.html')


# Get input image from client, predict class, and render respective .html page for solution
@app.route("/predict", methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        file = request.files['image']
        filename = file.filename
        print("@@ Input posted = ", filename)

        file_path = os.path.join('E:/Plant_Leaf_Disease_Prediction/Web_App/static/upload', filename)
        file.save(file_path)

        print("@@ Predicting class...")
        pred, output_page = pred_tomato_disease(tomato_plant=file_path)

        return render_template(output_page, pred_output=pred, user_image=file_path)


# For local system & cloud
if __name__ == "__main__":
    app.run(threaded=False, port=1010)
