import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Path to your model
MODEL_PATH = r"D:\waste_01\my_model.h5"

# Load model (cached)
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

model = load_model()

# 30 waste classes (in same order as training)
classes = [
    'aerosol_cans', 'aluminum_food_cans', 'aluminum_soda_cans', 'cardboard_boxes',
    'cardboard_packaging', 'clothing', 'coffee_grounds', 'disposable_plastic_cutlery',
    'eggshells', 'food_waste', 'glass_beverage_bottles', 'glass_cosmetic_containers',
    'glass_food_jars', 'magazines', 'newspaper', 'office_paper', 'paper_cups',
    'plastic_cup_lids', 'plastic_detergent_bottles', 'plastic_food_containers',
    'plastic_shopping_bags', 'plastic_soda_bottles', 'plastic_straws',
    'plastic_trash_bags', 'plastic_water_bottles', 'shoes', 'steel_food_cans',
    'styrofoam_cups', 'styrofoam_food_containers', 'tea_bags'
]

# Waste category mapping
waste_info = {
    "Recyclable": [
        'aerosol_cans', 'aluminum_food_cans', 'aluminum_soda_cans', 'cardboard_boxes',
        'cardboard_packaging', 'glass_beverage_bottles', 'glass_cosmetic_containers',
        'glass_food_jars', 'magazines', 'newspaper', 'office_paper', 'plastic_cup_lids',
        'plastic_detergent_bottles', 'plastic_food_containers', 'plastic_soda_bottles',
        'plastic_water_bottles', 'steel_food_cans'
    ],
    "Burnable": [
        'clothing', 'disposable_plastic_cutlery', 'plastic_shopping_bags',
        'plastic_straws', 'plastic_trash_bags', 'shoes', 'styrofoam_cups',
        'styrofoam_food_containers'
    ],
    "Compostable": [
        'coffee_grounds', 'eggshells', 'food_waste', 'tea_bags', 'paper_cups'
    ]
}

# Generate reverse mapping
category_details = {
    "Recyclable": "This item can be recycled. Make sure to rinse and dry it before placing it in the recycling bin.",
    "Burnable": "This item is burnable. Consider reuse or donation before incineration to reduce waste.",
    "Compostable": "This is compostable waste. You can compost it at home or in an industrial facility. Time to decompose varies from 1–6 months.",
    "Landfill": "This item should go to landfill if no other disposal method is available. Avoid contaminating other waste streams."
}

# UI Layout
st.set_page_config(page_title="Smart Waste Classifier", layout="centered")
st.title("🗑️ Smart Waste Classifier")
st.write("Upload an image of a waste item, and this app will predict the type and suggest how to dispose of it properly.")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

def preprocess_image(image):
    image = image.resize((224, 224))  # Ensure the same size as model input
    img_array = tf.keras.preprocessing.image.img_to_array(image)
    img_array = img_array / 255.0  # Normalize
    return np.expand_dims(img_array, axis=0)

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    st.write("🔍 Analyzing image...")

    input_tensor = preprocess_image(image)
    predictions = model.predict(input_tensor)
    pred_index = np.argmax(predictions)
    pred_class = classes[pred_index]
    confidence = predictions[0][pred_index] * 100

    # Find category
    if pred_class in waste_info["Recyclable"]:
        category = "Recyclable"
    elif pred_class in waste_info["Burnable"]:
        category = "Burnable"
    elif pred_class in waste_info["Compostable"]:
        category = "Compostable"
    else:
        category = "Landfill"

    # Display Results
    st.success(f"✅ Predicted: **{pred_class.replace('_', ' ').title()}**")
    st.info(f"📦 Category: **{category}**")
    st.write(f"🧠 Model Confidence: **{confidence:.2f}%**")
    st.markdown("---")
    st.subheader("♻️ Disposal Instructions")
    st.write(category_details[category])
