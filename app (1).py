import streamlit as st
import torch
import torchvision.transforms as transforms
import albumentations as A
from PIL import Image
import numpy as np
import segmentation_models_pytorch as smp

# Define the model architecture
model = smp.Unet(
    encoder_name="resnet50",
    encoder_weights="imagenet",
    in_channels=3,
    classes=1,
)

# Load the saved weights
model.load_state_dict(torch.load("checkpoints/best.pth", map_location=torch.device('cpu')))
model.eval()  # Set the model to evaluation mode

# Define image transformations
test_transforms = A.Compose([
    A.Resize(512, 512),
])

# Function to preprocess and predict on a single image
def predict_single_image(image, model, transforms):
    image = np.array(image)
    transformed = transforms(image=image)
    image = transformed['image']
    image = image / 255.0  # Normalize
    image = transforms.ToTensor()(image)
    image = image.unsqueeze(0)  # Add batch dimension

    with torch.no_grad():
        output = model(image)

    return output

# Streamlit app
st.title('Image Segmentation Prediction')

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Predict on the uploaded image
    if st.button('Predict'):
        with st.spinner('Predicting...'):
            output = predict_single_image(image, model, test_transforms)
            # Process the output (you might want to visualize or post-process it)
            # For simplicity, let's just display the raw output
            st.image(output.squeeze(0).cpu().numpy(), caption='Prediction', use_column_width=True)
