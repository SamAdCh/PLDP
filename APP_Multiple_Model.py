import streamlit as st
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
        'model_path': 'https://github.com/SamAdCh/PLDP/blob/master/tomato.h5',
        'threshold': 0.95,
        'classes': {
            0: ("Tomato - Bacteria Spot Disease", 'https://github.com/SamAdCh/PLDP/tree/master/templates/Tomato-Bacteria Spot.html'),
            1: ("Tomato - Early Blight Disease", 'https://github.com/SamAdCh/PLDP/tree/master/templates/Tomato-Early_Blight.html'),
            2: ("Tomato - Late Blight Disease", 'https://github.com/SamAdCh/PLDP/tree/master/templates/Tomato - Late_blight.html'),
            3: ("Tomato - Leaf Mold Disease", 'https://github.com/SamAdCh/PLDP/tree/master/templates/Tomato - Leaf_Mold.html'),
            4: ("Tomato - Septoria Leaf Spot Disease", 'https://github.com/SamAdCh/PLDP/tree/master/templates/Tomato - Septoria_leaf_spot.html'),
            5: ("Tomato - Two Spotted Spider Mite Disease", 'https://github.com/SamAdCh/PLDP/tree/master/templates/Tomato - Two-spotted_spider_mite.html'),
            6: ("Tomato - Target Spot Disease", 'https://github.com/SamAdCh/PLDP/tree/master/templates/Tomato - Target_Spot.html'),
            7: ("Tomato - Tomato Yellow Leaf Curl Virus Disease", 'https://github.com/SamAdCh/PLDP/tree/master/templates/Tomato - Tomato_Yellow_Leaf_Curl_Virus.html'),
            8: ("Tomato - Tomato Mosaic Virus Disease", 'https://github.com/SamAdCh/PLDP/tree/master/templates/Tomato - Tomato_mosaic_virus.html'),
            9: ("Tomato - Healthy and Fresh", 'https://github.com/SamAdCh/PLDP/tree/master/templates/Tomato-Healthy.html')
        }
    },
    'Corn': {
        'model_path': 'https://github.com/SamAdCh/PLDP/blob/master/corn.h5',
        'threshold': 0.8,
        'classes': {
            0: ("Corn - Blight Disease", 'https://github.com/SamAdCh/PLDP/tree/master/templates/Corn - Blight.html'),
            1: ("Corn - Common Rust Disease", 'https://github.com/SamAdCh/PLDP/tree/master/templates/Corn - Common_Rust.html'),
            2: ("Corn - Gray Leaf Spot Disease", 'https://github.com/SamAdCh/PLDP/tree/master/templates/Corn - Gray_Leaf_Spot.html'),
            3: ("Corn - Healthy and Fresh", 'https://github.com/SamAdCh/PLDP/tree/master/templates/Corn - Healthy.html')
        }
    },
    'Potato': {
        'model_path': 'https://github.com/SamAdCh/PLDP/blob/master/potato.h5',
        'threshold': 0.95,
        'classes': {
            0: ("Potato - Early Blight Disease", 'https://github.com/SamAdCh/PLDP/tree/master/templates/Potato_Early_Blight.html'),
            1: ("Potato - Healthy", 'https://github.com/SamAdCh/PLDP/tree/master/templates/Potato_Healthy.html'),
            2: ("Potato - Late Blight", 'https://github.com/SamAdCh/PLDP/tree/master/templates/Potato_Late_Blight.html')
        }
    },
    'Mango': {
        'model_path': 'https://github.com/SamAdCh/PLDP/blob/master/mango.h5',
        'threshold': 0.95,
        'classes': {
            0: ("Mango - Anthracnose Disease", 'https://github.com/SamAdCh/PLDP/tree/master/templates/Mango - Anthracnose.html'),
            1: ("Mango - Bacterial Canker Disease", 'https://github.com/SamAdCh/PLDP/tree/master/templates/Mango - Bacterial_Canker.html'),
            2: ("Mango - Cutting Weevil Disease", 'https://github.com/SamAdCh/PLDP/tree/master/templates/Mango - Cutting_Weevil.html'),
            3: ("Mango - Die Back Disease", 'https://github.com/SamAdCh/PLDP/tree/master/templates/Mango - Die_Back.html'),
            4: ("Mango - Gall Midge Disease", 'https://github.com/SamAdCh/PLDP/tree/master/templates/Mango - Gall_Midge.html'),
            5: ("Mango - Healthy", 'https://github.com/SamAdCh/PLDP/tree/master/templates/Mango - Healthy.html'),
            6: ("Mango - Powdery Mildew Disease", 'https://github.com/SamAdCh/PLDP/tree/master/templates/Mango - Powdery_Mildew.html'),
            7: ("Mango - Sooty Mould Disease", 'https://github.com/SamAdCh/PLDP/tree/master/templates/Mango - Sooty_Mould.html')
        }
    },
    'Pepper': {
        'model_path': 'https://github.com/SamAdCh/PLDP/blob/master/pepper.h5',
        'threshold': 0.05,
        'classes': {
            0: ("Pepper Bell Bacterial Spot", 'https://github.com/SamAdCh/PLDP/tree/master/templates/Pepper_Bell_Bacterial_spot.html'),
            1: ("Pepper Bell Healthy", 'https://github.com/SamAdCh/PLDP/tree/master/templates/Pepper_Bell_Healthy.html'),
            }
        },
    'Apple': {
        'model_path': 'https://github.com/SamAdCh/PLDP/blob/master/apple.h5',
        'threshold': 0.95,
        'classes': {
            0: ("Apple - Black Rot", 'https://github.com/SamAdCh/PLDP/tree/master/templates/Apple - Black_Rot.html'),
            1: ("Apple - Healthy", 'https://github.com/SamAdCh/PLDP/tree/master/templates/Apple - Healthy.html'),
            2: ("Apple - Scab", 'https://github.com/SamAdCh/PLDP/tree/master/templates/Apple - Scab.html'),
            3: ("Apple - Cedar Apple Rust", 'https://github.com/SamAdCh/PLDP/tree/master/templates/Apple - Cedar_Apple_Rust.html')
        }
    },
    'Strawberry': {
        'model_path': 'https://github.com/SamAdCh/PLDP/blob/master/strawberry.h5',
        'threshold': 0.95,
        'classes': {
            0: ("Strawberry - Healthy", 'https://github.com/SamAdCh/PLDP/tree/master/templates/Strawberry - Healthy.html'),
            1: ("Strawberry - Leaf Scorch Disease", 'https://github.com/SamAdCh/PLDP/tree/master/templates/Strawberry - Leaf_Scorch.html')
        }
    },
    'Rice': {
        'model_path': 'https://github.com/SamAdCh/PLDP/blob/master/rice.h5',
        'threshold': 0.8,
        'classes': {
            0: ("Rice - Brown Spot", 'https://github.com/SamAdCh/PLDP/tree/master/templates/Rice - BrownSpot.html'),
            1: ("Rice - Healthy", 'https://github.com/SamAdCh/PLDP/tree/master/templates/Rice - Healthy.html'),
            2: ("Rice - Hispa", 'https://github.com/SamAdCh/PLDP/tree/master/templates/Rice - Hispa.html'),
            3: ("Rice - Leaf Blast", 'https://github.com/SamAdCh/PLDP/tree/master/templates/Rice - LeafBlast.html')
        }
    }
    
}


# Load the models lazily when needed
models = {}

@st.cache(allow_output_mutation=True)
def load_models():
    for plant, config in plants.items():
        model_path = config['model_path']
        model = load_model(model_path)
        models[plant] = model
        st.write(f"Model for {plant} loaded successfully")

def get_model(plant_type):
    if plant_type in models:
        return models[plant_type]
    else:
        return None

def classify_image(image_path):
    # Load the classification model from C:/Desktop
    classification_model_path = 'DIRECT_URL_TO_CLASSIFICATION_MODEL'
    classification_model = load_model(classification_model_path)
    st.write("Classification model loaded successfully")

    # Load the image and resize it
    target_size = (128, 128)
    try:
        test_image = load_img(image_path, target_size=target_size)
        st.write("@@ Got Image for classification")
    except Exception as e:
        st.write("@@ Error loading image:", str(e))
        return "Unknown"

    # Convert the image to a numpy array and normalize it
    test_image = img_to_array(test_image) / 255
    test_image = np.expand_dims(test_image, axis=0)

    # Make predictions using the classification model
    result = classification_model.predict(test_image)
    st.write('@@ Classification result:', result)

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

def pred_disease(plant_type, image_path):
    model = get_model(plant_type)
    if model is None:
        return "Unknown Disease", 'https://github.com/SamAdCh/PLDP/tree/master/templates/unknown.html'

    # Load the image and resize it
    target_size = (128, 128)  # Default target size for Tomato, can be adjusted for other plant types
    if plant_type == 'Tomato':
        target_size = (256, 256)

    try:
        test_image = load_img(image_path, target_size=target_size)
        st.write("@@ Got Image for prediction")
    except Exception as e:
        st.write("@@ Error loading image:", str(e))
        return "Error loading image", 'error.html'

    # Convert the image to a numpy array and normalize it
    test_image = img_to_array(test_image) / 255
    test_image = np.expand_dims(test_image, axis=0)

    # Make predictions using the model
    result = model.predict(test_image)
    st.write('@@ Prediction result:', result)

    # Get the predicted class
    pred_class = np.argmax(result, axis=1)[0]  # Convert to scalar value

    if pred_class in plants[plant_type]['classes']:
        pred_label, template = plants[plant_type]['classes'][pred_class]
        return pred_label, template
    else:
        return "Unknown Disease", 'https://github.com/SamAdCh/PLDP/tree/master/templates/unknown.html'

def main():
    # Load the models
    load_models()

    # Define the Streamlit app layout
    st.title("Plant Disease Classifier")

    plant_type = st.selectbox("Select Plant Type", list(plants.keys()))

    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        # Save the uploaded image to a temporary file
        with open(os.path.join("temp", uploaded_file.name), "wb") as f:
            f.write(uploaded_file.getbuffer())

        image_path = os.path.join("temp", uploaded_file.name)
        st.image(image_path, caption="Uploaded Image", use_column_width=True)

        if st.button("Classify"):
            classification_result = classify_image(image_path)
            st.write("Classification Result:", classification_result)

            disease, template = pred_disease(plant_type, image_path)
            st.write("Predicted Disease:", disease)

            # Display the HTML template for the predicted disease
            template_path = os.path.join("templates", template)
            with open(template_path, "r") as f:
                template_content = f.read()
            st.markdown(template_content, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
