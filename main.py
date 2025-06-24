import streamlit as st
import numpy as np
import random
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model('skin_cancer_cnn.h5')

malignant_precautions = [
    "Consult a dermatologist immediately for further evaluation.",
    "Avoid direct sun exposure and always wear sunscreen with SPF 50+.",
    "Monitor for any changes in size, shape, or color of the lesion.",
    "Schedule a biopsy to confirm the diagnosis and discuss treatment options.",
    "Keep the affected area covered and avoid scratching or irritation.",
]

# List of safe statements for benign cases
benign_safe_statements = [
    "The lesion appears benign, but regular self-checks are still recommended.",
    "Maintain healthy skin by using moisturizer and staying hydrated.",
    "Continue protecting your skin from excessive sun exposure.",
    "If you notice any changes in the future, consult a dermatologist.",
    "Your skin looks healthy! Keep up your skincare routine.",
]


# Function to preprocess and predict the image
def predict_skin_cancer(image_path, model):
    img = image.load_img(image_path, target_size=(224, 224))  # Load Image
    img_array = image.img_to_array(img) / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    # Make Prediction
    prediction = model.predict(img_array)
    class_label = "Malignant" if prediction > 0.5 else "Benign"

    return class_label, img


# Streamlit App
st.title("AN EFFECTIVE APPROACH TO DETECT SKIN CANCER USING CNN")

st.markdown("""
    This application aids in identifying skin cancer by analyzing uploaded images. Using a deep learning model, it predicts whether a given skin lesion is Malignant or Benign.
""")

# File uploader
uploaded_image = st.file_uploader("Drop your image here...", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    # Predict and display results
    class_label, img = predict_skin_cancer(uploaded_image, model)

    # Display the result image with the prediction title
    st.image(uploaded_image, caption="Uploaded Image")
  # Show the uploaded image first
    st.write(f"**Prediction: {class_label}**")
    if class_label == "Malignant":
        st.error("⚠️ **Precautions to Take:**")
        for precaution in random.sample(malignant_precautions, 4):  # Select 4 random precautions
            st.write(f"- {precaution}")
    else:
        st.success("✅ **Safe Skin Care Tips:**")
        for safe_statement in random.sample(benign_safe_statements, 3):  # Select 3 random safe statements
            st.write(f"- {safe_statement}")

    # # Display processed result image with the prediction title
    # fig, ax = plt.subplots()
    # ax.imshow(img)
    # ax.set_title(f"Predicted: {class_label}")
    # ax.axis("off")
    # st.pyplot(fig)

# Additional info and styling
st.markdown("""
    ### About the Model:
    This system is built using a Convolutional Neural Network (CNN), a deep learning architecture widely used for image analysis. The model has been trained on a large dataset of skin lesion images, enabling it to distinguish between Benign (non-cancerous) and Malignant (cancerous) lesions with high accuracy.

    #### Features:
    - **Input**: Images of skin lesions
    - **Output**: Classification as either **Benign** or **Malignant**
    - **Fast & Accurate**: Utilizes deep learning for rapid and precise analysis
    - **User-Friendly Interface**: Simple drag-and-drop functionality for image uploads

    #### How to use:
    1. Upload an image of a skin lesion.
    2. The model will predict if it's **Benign** or **Malignant**.
""")
