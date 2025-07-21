import streamlit as st
import torch
import torch.nn as nn
from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights
from PIL import Image
import gdown
import os

# Update model link and name
# MODEL_URL = "https://drive.google.com/uc?id=1WjT7Qlwm5fh2Gs9o8hwwKljo1eX5Cuj5"  # 🔁 replace with your new Drive ID ###ah19
#MODEL_URL = "https://drive.google.com/uc?id=1lGUbaudev71JDMJdTh829RAScQARH0Pi"  # 🔁 replace with your new Drive ID ###ih08
# https://drive.google.com/file/d/1lGUbaudev71JDMJdTh829RAScQARH0Pi/view?usp=sharing
#MODEL_FILENAME = "convnext_model_1000.pth"

MODEL_URL = "https://drive.google.com/uc?id=1nMmfKyNADwkiYlCEiyJH7WK3ResO223Z"  # 🔁 replace with your new Drive ID ###ih08
#https://drive.google.com/file/d/1nMmfKyNADwkiYlCEiyJH7WK3ResO223Z/view?usp=sharing
MODEL_FILENAME = "convnext_model2_1000.pth"

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_FILENAME):
        gdown.download(MODEL_URL, MODEL_FILENAME, quiet=False)

    # Load pretrained base model
    model = convnext_tiny(weights=ConvNeXt_Tiny_Weights.DEFAULT)
    num_ftrs = model.classifier[2].in_features
    model.classifier[2] = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(num_ftrs, 8)
    )

    # Load checkpoint
    checkpoint = torch.load(MODEL_FILENAME, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Get class names
    class_to_idx = checkpoint.get('class_to_idx', {
        'A+': 0, 'A-': 1, 'AB+': 2, 'AB-': 3,
        'B+': 4, 'B-': 5, 'O+': 6, 'O-': 7
    })
    idx_to_class = {v: k for k, v in class_to_idx.items()}

    return model, idx_to_class

# Load model once (cached)
model, idx_to_class = load_model()

#st.title("🩸 Blood Group Prediction (ConvNeXt Tiny)")
# st.title("🩸 Blood Group Prediction (used ")
import streamlit as st

st.title("🩸 Blood Group Prediction")

# Small font details below the title
st.markdown(
    """
    <p style='font-size: 14px; color: white;'>
        Used ConvNeXt-Tiny <br>
        Total images: 8000 <br>
        Number of training images: 6400 <br>
        Number of testing images: 1600
    </p>
    """,
    unsafe_allow_html=True
)


uploaded_file = st.file_uploader("Upload a blood image", type=["jpg", "jpeg", "png", "bmp", "svg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    transform = ConvNeXt_Tiny_Weights.DEFAULT.transforms()
    from torchvision import transforms

    # transform = transforms.Compose([
    #     transforms.Resize((224, 224)),
    #     transforms.ToTensor(),
    #     transforms.Normalize([0.485, 0.456, 0.406],
    #                          [0.229, 0.224, 0.225])
    # ])

    input_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(input_tensor)
        _, predicted = torch.max(outputs, 1)
        predicted_label = idx_to_class[predicted.item()]
        st.success(f"Predicted Blood Group: **{predicted_label}**")
