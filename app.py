import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np

# Configuration
MODEL_PATH = "solar_panel_classifier.pth"
CLASS_NAMES = ['Bird-drop', 'Clean', 'Dusty', 'Electrical-damage', 'Physical-Damage', 'Snow-Covered']
IMAGE_SIZE = 224

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
def load_model():
    model = models.resnet18(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, len(CLASS_NAMES))
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()
    return model

# Transform image
def transform_image(image):
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0)  # add batch dimension

# Predict class
def predict(model, image_tensor):
    image_tensor = image_tensor.to(device)
    with torch.no_grad():
        outputs = model(image_tensor)
        _, predicted = torch.max(outputs, 1)
        return CLASS_NAMES[predicted.item()]

# Maintenance recommendation
def get_recommendation(label):
    recommendations = {
        "Clean": "No action needed.",
        "Dusty": "Recommend cleaning panel soon.",
        "Bird-drop": "Clean off bird droppings to avoid hotspots.",
        "Electrical-damage": "Call technician for electrical inspection.",
        "Physical-Damage": "Urgent: Replace or repair damaged panel.",
        "Snow-Covered": "Clear snow for optimal performance."
    }
    return recommendations.get(label, "No recommendation available.")

# Streamlit UI
st.title("🔋 Solar Panel Condition Classifier")
st.write("Upload an image of a solar panel to detect its condition.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Uploaded Image', use_column_width=True)

    with st.spinner('Classifying...'):
        model = load_model()
        image_tensor = transform_image(image)
        prediction = predict(model, image_tensor)
        recommendation = get_recommendation(prediction)

    st.success(f"Prediction: **{prediction}**")
    st.info(f"Recommendation: {recommendation}")
